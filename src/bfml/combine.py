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
    datasets: List[DatasetType], probabilities: Optional[List[float]] = None, seed: Optional[int] = None, maximum_length=None, long=False
) -> DatasetType:

    from datasets.arrow_dataset import Dataset
    from datasets.iterable_dataset import IterableDataset

    if any(len(dset) > 5_000_000 for dset in datasets):
        logging.warning('Trying to interleave a large dataset -- this may lead to hashing errors or slowdowns! Pass long=True to supress this warning.')
        if not long:
            sys.exit(1)

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
        return _interleave_map_style_datasets_continuous(datasets, probabilities, seed, maximum_length)
    else:
        raise NotImplementedError


def _interleave_map_style_datasets_continuous(
    datasets: List["Dataset"],
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    maximum_length = None,
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
    repeated_datasets = [[] for _ in range(len(datasets))]
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
            # we ran out of examples, so we reset the current index and keep track of when it happened
            if current_index[source_idx] >= lengths[source_idx]:
                repeated_datasets[source_idx].append(len(indices))
                current_index[source_idx] = 0

                if all(len(i) > 0 for i in repeated_datasets):
                    break

            if maximum_length and len(indices) > maximum_length:
                break

            if len(indices) % 1000000 == 0:
                print(len(indices))


            # let's add the example at the current index of the `source_idx`-th dataset
            indices.append(current_index[source_idx] + offsets[source_idx])
            current_index[source_idx] += 1
        print('done selecting')
    return concatenated_datasets, indices, repeated_datasets


class SimpleIterableDataset(_BaseExamplesIterable):

    def __init__(self, dataset):
        """
            Convert a huggingface dataset to an IterableDataset. Probably won't work very well :)
        :param dataset:
        """

        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)