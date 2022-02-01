import math, random, logging, numpy as np
from pprint import pformat
from torch.utils.data.sampler import Sampler, RandomSampler


class MultilingualBatchSampler(Sampler):

    def __init__(self, training_data, language_probabilities, config):
        """
         Older implementation of batch sampler. Works with get_sampler defined below.
        :param training_data:
        :param language_probabilities:
        :param config:
        """

        self.training_samplers = {}
        self.training_iterables = {}
        self.language_offsets = {}
        self.languages = []
        self.probabilities = []
        self.examples = [v[0] for k,v in training_data.items()]

        self.ta = config['training_arguments']

        #Batch size here does not include gradient accumulation steps, as that is handled in the training loop
        self.batch_size = self.ta['per_device_train_batch_size']  * config['n_gpu']

        running_total = 0
        for k,v in training_data.items():

            #Create samplers and starting iterators for each language
            self.training_samplers[k] = RandomSampler(v)
            self.training_iterables[k] = self.training_samplers[k].__iter__()

            #Calculate where the first example for each language will be in the concatenated dataset
            self.language_offsets[k] = running_total
            running_total += len(v)

            #Keep track of language order corresponding to offsets
            self.languages.append(k)
            self.probabilities.append(language_probabilities[k])

        logging.info('Language offsets: {}'.format(self.language_offsets))
        logging.info('Total number of examples across all languages: {}'.format(running_total))

        self.lang2id = {lang:i for i,lang in enumerate(self.languages)}
        logging.info('Lang2id: {}'.format(self.lang2id))


    def __iter__(self):


        batches = []
        total_number_batches = self.ta['max_steps'] // self.batch_size

        for i in range(total_number_batches):
            batch_language = random.choices(self.languages, weights=self.probabilities, k=1)[0]
            batch = []

            for i in range(self.batch_size):
                try:
                    #Absolute index = relative index from sampler + language offset
                    batch.append(self.training_iterables[batch_language].__next__() + self.language_offsets[batch_language])
                except StopIteration:
                    #Finished all batches from a language. Reset iterator and continue filling the partial batch
                    self.training_iterables[batch_language] = self.training_samplers[batch_language].__iter__()
                    batch.append(self.training_iterables[batch_language].__next__() + self.language_offsets[batch_language])
            batches.append(batch)


        return iter(batches)

    def __len__(self):
        return self.ta['max_steps'] // self.batch_size

def get_sampler(training_data, language_set, config):
    total_examples = 0

    for k,v in training_data.items():
        total_examples += len(v)

    language_probabilities = {}

    total_probability = 0.0
    for k,v in training_data.items():
        language_probability = len(v)/total_examples
        exponentiated_probability = math.pow(language_probability, config['dataset_settings']['alpha'])
        total_probability += exponentiated_probability
        language_probabilities[k] = exponentiated_probability

    #Re-normalize probabilites
    for k,v in language_probabilities.items():
        language_probabilities[k] = v/total_probability

    logging.info('Created multilingual weighted sampler.')
    logging.info(pformat(language_probabilities))

    sampler = MultilingualBatchSampler(training_data, language_probabilities, config)

    #Concatante data from all languages into a single dataset, in the order expected from the sampler
    flattened_training_data = []
    language_ids = []
    for language in sampler.languages:
        flattened_training_data.extend(training_data[language])
        language_ids_per_example = [sampler.lang2id[language]] * len(training_data[language])
        language_ids.extend(language_ids_per_example)

    return flattened_training_data, language_ids, sampler

def get_language_probabilities(lengths, alpha=0.3):

    language_probabilities = {}

    total_examples = sum(lengths.values())

    total_probability = 0.0
    for lang, length in lengths.items():
        language_probability = length/total_examples
        exponentiated_probability = language_probability ** alpha
        total_probability += exponentiated_probability
        language_probabilities[lang] = exponentiated_probability

    #Re-normalize probabilites
    language_probabilities = {lang: probability/total_probability for lang, probability in language_probabilities.items()}

    logging.info('Created language probabilities.')
    logging.info(pformat(language_probabilities))

    return language_probabilities

class WeightedMultiSourceSampler(Sampler):

    def __init__(self, training_data, probabilities, seed, min_repeat_val=1, max_batches=None):
        """
            Batch Sampler used for mixing multiple sources of data according to probabilities. If a source is finished
            before other sources, it will be cycled until all sources have been finished at least once. Pregenerates
            all batches before returning. Expects final dataset to be concatenated in the order of self.languages.

        :param training_data: Dictionary of source_name: source_data pairs
        :param probabilities: Dictionary of source_name: source_probability pairs, used for weighed sampling
        :param seed: Seed for deciding order of batches
        :param min_repeat_val: Minimum number of times to repeat each source before breaking
        """

        self.training_samplers = {}
        self.training_iterables = {}
        self.language_offsets = {}
        self.languages = []
        self.probabilities = []
        self.num_examples = {}
        self.seed = seed
        self.min_repeat_val = min_repeat_val
        self.max_batches = max_batches

        #Batch size will be set by Trainer during training loop
        self.batch_size = None

        running_total = 0
        for k,v in training_data.items():

            #Create samplers and starting iterators for each language
            self.training_samplers[k] = RandomSampler(v)
            self.training_iterables[k] = self.training_samplers[k].__iter__()

            #Calculate where the first example for each language will be in the concatenated dataset
            self.language_offsets[k] = running_total
            running_total += len(v)

            #Keep track of language order corresponding to offsets
            self.languages.append(k)
            self.probabilities.append(probabilities[k])
            self.num_examples[k] = len(v)

        logging.info('Language ordering: {}'.format(self.languages))
        logging.info('Language offsets: {}'.format(self.language_offsets))
        logging.info('Examples per language: {}'.format(self.num_examples))
        logging.info('Total number of examples across all languages: {}'.format(running_total))

        self.lang2id = {lang:i for i,lang in enumerate(self.languages)}
        logging.info('Lang2id: {}'.format(self.lang2id))


    def __iter__(self):

        if self.batch_size is None:
            logging.error('Trying to generate batches without batch size! Set self.batch_size before calling this function')
            raise AssertionError

        def iter_random_indices(seed):
            """ Get an infinite iterator that randomly samples the index of the source to pick examples from.
                Taken from HF Datasets combine.py
            """
            rng = np.random.default_rng(seed)
            while True:
                yield from (i for i in rng.choice(self.languages, size=1000, p=self.probabilities))


        batches = []
        self.repeats = [0 for _ in range(len(self.languages))] #keep track of how many times the language has been repeated


        for language_idx in iter_random_indices(self.seed):

            if all(lang_repeat >= self.min_repeat_val for lang_repeat in self.repeats) or (self.max_batches and len(batches) >= self.max_batches): #break out of loop once we have seen all examples from every language
                break

            batch = []

            for i in range(self.batch_size):
                try:
                    #Absolute index = relative index from sampler + language offset
                    batch.append(self.training_iterables[language_idx].__next__() + self.language_offsets[language_idx])
                except StopIteration:
                    #Finished all batches from a language. Reset iterator and continue filling the partial batch
                    self.repeats[self.lang2id[language_idx]] += 1
                    self.training_iterables[language_idx] = self.training_samplers[language_idx].__iter__()
                    batch.append(self.training_iterables[language_idx].__next__() + self.language_offsets[language_idx])
            batches.append(batch)

        logging.info('Done generating batches. Repeats: {} Languages: {} Total Batches Generated: {}'.format(self.repeats,self.languages, len(batches)))
        return iter(batches)



