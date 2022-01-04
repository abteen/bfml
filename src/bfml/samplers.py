import math, random, logging
from pprint import pformat
from torch.utils.data.sampler import Sampler, RandomSampler


class MultilingualBatchSampler(Sampler):

    def __init__(self, training_data, language_probabilities, config):

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

