import torch, logging, random, jsonlines, pandas, sys
from tqdm import trange

class PretrainingDatasetBase(torch.utils.data.Dataset):

    def __init__(self, training_data, config):

        if isinstance(training_data, list):
            self.examples = training_data
        elif isinstance(training_data, str):
            logging.info('Training data was given as a string -- assuming this is a file and reading line by line.')
            logging.info('Given string: {}'.format(training_data))
            self.examples = []
            with open(training_data, 'r') as f:
                for line in f:
                    self.examples.append(line.strip())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):

        raise NotImplementedError

class PretrainingDataset(PretrainingDatasetBase):

    def __init__(self, training_data, tokenizer, **config):

        super().__init__(training_data, **config)

        self.tokenizer = tokenizer
        self.tokenization_settings = config['tokenizer_settings']

        print(self.tokenization_settings)

    def __getitem__(self, idx):

        instance = self.examples[idx]

        enc = self.tokenizer(
            instance,
            **self.tokenization_settings['tokenization']
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0)
        }

class PretrainingDatasetWithLanguageEmb(PretrainingDatasetBase):

    def __init__(self, training_data, language_ids, tokenizer, config):

        super().__init__(training_data, config)

        self.tokenizer = tokenizer

        if isinstance(language_ids, list):
            assert len(language_ids) == len(self.examples)
            self.language_ids = language_ids
        elif isinstance(language_ids, dict):
            logging.info('language ids given as dict -- setting all ids to {}'.format(config['target_language']))
            self.language_ids = [language_ids[config['target_language']]] * len(self.examples)
        else:
            logging.error('Unknown input type {} for language_ids, expected list or dict'.format(type(language_ids)))
            sys.exit(0)

        self.tokenization_settings = config['tokenizer_settings']

        logging.info('Given tokenization settings: {}'.format(self.tokenization_settings))

    def __getitem__(self, idx):

        instance = self.examples[idx]
        language_id = self.language_ids[idx]

        enc = self.tokenizer(
            instance,
            **self.tokenization_settings['tokenization']
        )

        input_ids = enc['input_ids'].squeeze(0)

        #Taken from huggingface docs 12/7
        language_ids = torch.tensor([language_id] * input_ids.shape[0])

        return {
            'input_ids': input_ids,
            'attention_mask': enc['attention_mask'].squeeze(0),
            'language_ids': language_ids
        }


class xnliTSVDataset(torch.utils.data.Dataset):
    def __init__(self, file, tokenizer, max_len, lang, format, pred_loop_key=None, unseen=False):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang = lang
        self.unseen = unseen
        self.format = format

        if pred_loop_key is None:
            self.pred_loop_key = self.lang
        else:
            self.pred_loop_key = pred_loop_key

        self.examples = self.read_file(file)

        for i,ex in enumerate(self.examples[:3]):
            logging.info('Example {} for {} lang: {}'.format(i,lang,ex))

        logging.info('Pred loop key: {}'.format(self.pred_loop_key))
        logging.info('Number of examples: {}'.format(len(self.examples)))
        logging.info('Data loaded from: {}'.format(file))


        random.shuffle(self.examples)


        logging.info('----------------------------------')


    def __getitem__(self, idx):
        # return self.tokenized_examples[idx]
        return self.encode(idx)

    def __len__(self):
        return len(self.examples)

    def read_file(self, file):

        inps = []
        labels_found = set()

        label2id = {
            'contradiction' : 0,
            'contradictory' : 0,
            'neutral' : 1,
            'entailment' : 2
        }


        with open(file, 'r') as f:
            for i, line in enumerate(f.readlines()):

                if i == 0:
                    print(line)
                    continue

                split = line.strip().split('\t')

                #Data input format: (premise, hypothesis, label)

                if self.format == 'mnli': #multinli_1.0_train.txt
                    inps.append((split[5], split[6], label2id[split[0]]))
                    labels_found.add(split[0])
                elif self.format == 'xnli': #xnli.dev.tsv, xnli.test.tsv
                    if split[0] == self.lang:
                        inps.append((split[6], split[7], label2id[split[1]]))
                        labels_found.add(split[1])
                elif self.format == 'mnli_translated': #mnli.train.es.tsv
                    inps.append((split[0], split[1], label2id[split[2]]))
                    labels_found.add(split[2])
                elif self.format == 'anli':
                    if split[1] == self.lang:
                        inps.append((split[2], split[3], label2id[split[4]]))
                        labels_found.add(split[4])
                elif self.format in ['translate-train']: #xnli_unseen.dev.tsv, translate_train.dev.tsv, train/${lang}.tsv
                    if split[0] == self.lang:
                        inps.append((split[1], split[2], label2id[split[3]]))
                        labels_found.add(split[3])



        return inps


    def encode(self, id):
        instance = self.examples[id]

        s1 = instance[0]
        s2 = instance[1]
        label = instance[2]

        enc = self.tokenizer(
            s1,s2,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids= True,
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'labels': label
        }

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, file, lang, max_len, tokenizer):

        self.tokenizer = tokenizer

        self.max_len = max_len
        self.lang = lang

        self.create_label2id()

        self.examples = self.read_file(file)

        logging.info('Loaded {} examples for {}'.format(len(self.examples), lang))


    def __getitem__(self, idx):
        return self.encode(idx)

    def __len__(self):
        return len(self.examples)

    def create_label2id(self):

        ner_tags = [
            'B-ORG',
            'I-ORG',
            'B-PER',
            'I-PER',
            'B-MISC',
            'I-MISC',
            'B-LOC',
            'I-LOC',
            'O'
        ]

        iter = 0
        self.label2id = {}
        for tag in ner_tags:
            self.label2id[tag] = iter
            iter += 1

    def read_file(self, file, convert_labels=True):

        inps = []

        with open(file, 'r') as f:
            temp_tokens = []
            temp_labels = []
            for line in f:
                if line.strip():

                    token = line.strip().split('\t')
                    assert len(token) == 2

                    if convert_labels:
                        temp_tokens.append(token[0].replace(self.lang + ':', ''))
                        temp_labels.append(self.label2id[token[1]])

                    else:
                        temp_tokens.append(token[0].replace(self.lang + ':', ''))
                        temp_labels.append(token[1])

                else:
                    inps.append((temp_tokens,temp_labels))
                    temp_tokens = []
                    temp_labels = []

        tokenized_examples = []

        for instance in inps:
            forms = instance[0]
            labels = instance[1]

            expanded_labels = []

            for i in range(0, len(forms)):

                subwords = self.tokenizer.tokenize(forms[i])

                for j in range(0, len(subwords) - 1):
                    expanded_labels.append(-100)
                expanded_labels.append(labels[i])

            s1 = ' '.join(forms)

            enc = self.tokenizer(
                s1,
                max_length=self.max_len,
                truncation=True,
                return_token_type_ids=True,
            )

            if len(expanded_labels) > self.max_len:
                expanded_labels = expanded_labels[:self.max_len]

            enc['labels'] = expanded_labels
            tokenized_examples.append(enc)

        return tokenized_examples

    def encode(self, id):

        return self.examples[id]

class NERDatasetLangaugeEmbedding(torch.utils.data.Dataset):
    def __init__(self, file, lang, max_len, tokenizer):

        self.tokenizer = tokenizer

        self.max_len = max_len
        self.lang = lang

        # self.lang2id = {'sw': 0, 'ay': 1, 'en': 2, 'es': 3}
        # self.langid = self.lang2id[lang]

        self.langid = 2

        self.create_label2id()

        self.examples = self.read_file(file)

        logging.info('Loaded {} examples for {}'.format(len(self.examples), lang))


    def __getitem__(self, idx):
        return self.encode(idx)

    def __len__(self):
        return len(self.examples)

    def create_label2id(self):

        ner_tags = [
            'B-ORG',
            'I-ORG',
            'B-PER',
            'I-PER',
            'B-MISC',
            'I-MISC',
            'B-LOC',
            'I-LOC',
            'O'
        ]

        iter = 0
        self.label2id = {}
        for tag in ner_tags:
            self.label2id[tag] = iter
            iter += 1

    def read_file(self, file, convert_labels=True):

        inps = []

        with open(file, 'r') as f:
            temp_tokens = []
            temp_labels = []
            for line in f:
                if line.strip():

                    token = line.strip().split('\t')
                    assert len(token) == 2

                    if convert_labels:
                        temp_tokens.append(token[0].replace(self.lang + ':', ''))
                        temp_labels.append(self.label2id[token[1]])

                    else:
                        temp_tokens.append(token[0].replace(self.lang + ':', ''))
                        temp_labels.append(token[1])

                else:
                    inps.append((temp_tokens,temp_labels))
                    temp_tokens = []
                    temp_labels = []

        tokenized_examples = []

        for instance in inps:
            forms = instance[0]
            labels = instance[1]

            expanded_labels = []

            for i in range(0, len(forms)):

                subwords = self.tokenizer.tokenize(forms[i])

                for j in range(0, len(subwords) - 1):
                    expanded_labels.append(-100)
                expanded_labels.append(labels[i])

            s1 = ' '.join(forms)



            enc = self.tokenizer(
                s1,
                max_length=self.max_len,
                truncation=True,
                return_token_type_ids=True,
            )

            if len(expanded_labels) > self.max_len:
                expanded_labels = expanded_labels[:self.max_len]

            enc['labels'] = expanded_labels
            language_ids = [self.langid] * len(enc['input_ids'])
            enc['language_ids'] = language_ids

            tokenized_examples.append(enc)

        return tokenized_examples

    def encode(self, id):

        return self.examples[id]

