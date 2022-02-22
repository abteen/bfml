from typing import TYPE_CHECKING, Any, List, Optional, TypeVar

import numpy as np, sys

from collections import defaultdict

from datasets.info import DatasetInfo
from datasets.iterable_dataset import _BaseExamplesIterable
import logging


if TYPE_CHECKING:
    from datasets.arrow_dataset import Dataset
    from datasets.iterable_dataset import IterableDataset


DatasetType = TypeVar("DatasetType", "Dataset", "IterableDataset")


def interleave_datasets_cycled(
    datasets: List[DatasetType], languages=None, probabilities: Optional[List[float]] = None, seed: Optional[int] = None, batch_size=1, min_repeat_val=1, maximum_length=None,
) -> DatasetType:

    from datasets.arrow_dataset import Dataset
    from datasets.iterable_dataset import IterableDataset

    if any(len(dset) > 1_000_000 for dset in datasets):
        logging.warning('Trying to interleave a large dataset -- a custom fingerprint will be used to prevent hashing errors')

    if not datasets:
        raise ValueError("Unable to interleave an empty list of datasets.")
    iterable = isinstance(datasets[0], IterableDataset)
    map_style = isinstance(datasets[0], Dataset)
    if not (iterable ^ map_style):
        raise ValueError(
            f"Expected a list Dataset objects or a list of IterableDataset objects, but first element is a {type(datasets[0])}"
        )
    for dataset in datasets[1:]:
        if (map_style and not isinstance(dataset, Dataset)) or (iterable and not isinstance(dataset, IterableDataset)):
            raise ValueError(
                f"Unable to interleave a {type(datasets[0])} with a {type(dataset)}. Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    if map_style:
        return _interleave_map_style_datasets_continuous(datasets=datasets, languages=languages, probabilities=probabilities, seed=seed, batch_size=batch_size, min_repeat_val=min_repeat_val, maximum_length=maximum_length)
    else:
        raise NotImplementedError


def _interleave_map_style_datasets_continuous(
    datasets: List["Dataset"],
    languages=None,
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    batch_size = 1,
    min_repeat_val = 1,
    maximum_length=None,
    info: Optional[Any] = None,
    split: Optional[Any] = None,
    **kwargs,
) -> "Dataset":
    """

    Taken from huggingface datasets, and modified such that it finishes once the largest dataset has been completely cycled once.

    Interleave several map-style datasets (sources) into a single map-style dataset.
    The new dataset is constructed by alternating between the sources to get the examples.
    If `probabilities = None` (default) the new dataset is constructed by cycling between each source to get the examples.
    If `probabilities` is not `None, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.
    Args:
        datasets (:obj:`List[Dataset]`): list of datasets to interleave
        probabilities (:obj:`List[float]`, optional, default None): If specified, the new dataset is constructued by sampling
            examples from one source at a time according to these probabilities.
        seed (:obj:`int`, optional, default None): The random seed used to choose a source for each example.
        **kwargs: Keyword arguments to be passed to :meth:`datasets.Datasets.select` when selecting the indices used to interleave the datasets.
    Output:
        :class:`datasets.Dataset`
    """
    from datasets.arrow_dataset import concatenate_datasets

    # To interleave the datasets, we concatenate them and then we re-order the indices
    concatenated_datasets = concatenate_datasets(datasets, info=info, split=split)

    # Let's now build the indices to pass to .select()
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])
    repeats = [0 for _ in range(len(lengths))]

    if batch_size == 0:
        logging.warning('Generating batches using a batch size of 0! Setting to 1 -- interleaving datasets by example')
        batch_size = 1

    max_examples = maximum_length * batch_size if maximum_length else None

    if probabilities is None:
        # Example: If lengths of the datasets are [3, 4, 5]
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 6, 9]
        # Note that we only have 3 examples per dataset since the first dataset ran out of examples
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape(-1, 1)).flatten().tolist()
    else:

        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(datasets), size=1000, p=probabilities))

        current_index = [0] * len(datasets)
        indices = []
        for source_idx in iter_random_indices():

            if all(i >= min_repeat_val for i in repeats) or (max_examples and len(indices) > max_examples):
                break

            for _ in range(batch_size):
                if current_index[source_idx] >= lengths[source_idx]:
                    repeats[source_idx] += 1
                    current_index[source_idx] = 0

                # let's add the example at the current index of the `source_idx`-th dataset
                indices.append(current_index[source_idx] + offsets[source_idx])
                current_index[source_idx] += 1

    file_name = '_'.join(languages) + '_' +  '_'.join([str(lng) for lng in lengths]) + '_' + str(seed)
    logging.info('Languages: {} Repeats: {} Total Example: {} Generated fingerprint: {}'.format(
        languages,
        repeats,
        len(indices),
        file_name
    ))

    return concatenated_datasets.select(indices, new_fingerprint=file_name, **kwargs)


class SimpleIterableDataset(_BaseExamplesIterable):

    def __init__(self, dataset):
        """
            Convert a huggingface dataset to an IterableDataset. Probably won't work very well :)
        :param dataset:
        """

        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)