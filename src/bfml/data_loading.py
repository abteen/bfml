import torch, numpy as np, os, logging
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict


def read_lines(file, preserve_lines=True):
    """

    :param file: File to read from line-by-line
    :param preserve_lines: If True, return a list n items, each corresponding to a line.
                        Else, return a list of 1 item consisting of all lines joined together
    :return: List[str]
    """

    file_lines = []
    with open(file, 'r') as f:
        for line in f:
            file_lines.append(line.strip())

    if not preserve_lines:
        collapsed_document = ' '.join(line for line in file_lines)
        return [collapsed_document]

    return file_lines

def chunk_and_tokenize_documents(documents, tokenizer, max_length, dataset=True):
    """

    :param documents: List[String] where each element is a document
    :param tokenizer: tokenizer
    :param max_length: length of sequences which will be inputted into the model
    :return: input_ids and attention masks generated by chunking each document into tensors of size max_length.
                No overlap between tensors. Does not respect sentence boundaries, but respects document boundaries.
    """

    bos_token = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer.eos_token_id], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer.pad_token_id], dtype=torch.int64)
    zero_token = torch.tensor([0], dtype=torch.int64) #In case pad token != 0
    double_1 = torch.tensor([1,1], dtype=torch.int64)

    max_len = max_length - 2

    final_input_ids = []
    attention_masks = []

    for document in documents:

        a = tokenizer(document, return_tensors='pt', add_special_tokens=False)

        input_ids = torch.split(a['input_ids'][0], max_len)
        attention_mask = torch.split(a['attention_mask'][0], max_len)

        assert len(input_ids) == len(attention_mask)

        for i in range(len(input_ids)):

            inp = torch.cat([bos_token, input_ids[i], eos_token])
            attn = torch.cat((attention_mask[i], double_1))

            if len(inp) < max_length:
                pad_len = max_length-inp.shape[0]
                inp = torch.cat([inp, pad_token.expand(pad_len)])
                attn = torch.cat([attn, zero_token.expand(pad_len)])

            final_input_ids.append(inp)
            attention_masks.append(attn)

    if dataset:
        return {
            'input_ids' : final_input_ids,
            'attention_mask' : attention_masks
        }

    stacked_input_ids = torch.stack(final_input_ids)
    stacked_attentions = torch.stack(attention_masks)


    return stacked_input_ids, stacked_attentions

def tokenize_documents(documents, tokenizer, max_length, dataset=True, padding='max_length'):
    """

    Similar to chunk_and_tokenize_documents but does no chunking. Helpful when using models which expect
    data in a different format as the chunking assumes. Truncates data to the given maximum length. Processed data
    retains all information returned by the tokenizer call.

    :param documents:
    :param tokenizer:
    :param max_length:
    :param dataset:
    :return:
    """

    tokenized = []

    for document in documents:

        tokenized.append(
            tokenizer(document, return_tensors='pt', add_special_tokens=True, truncation=True, max_length=max_length, padding=padding)
        )

    if dataset:

        #Convert the list of dictionaries to a dictionary of lists
        # Tokenizer output is a list of lists; since we are only tokenizing one document/example at a time, we squeeze to flatten the list
        dataset_dict = {k : [tokenizer_output[k].squeeze() for tokenizer_output in tokenized] for k in tokenized[0].keys()}

        return dataset_dict

    return tokenized


def chunk_and_tokenize_documents_for_dataset_np(documents, tokenizer, max_length):
    """
        Numpy implementation
    :param documents: List[String] where each element is a document
    :param tokenizer: tokenizer
    :param max_length: length of sequences which will be inputted into the model
    :return: input_ids and attention masks generated by chunking each document into tensors of size max_length.
                No overlap between tensors. Does not respect sentence boundaries, but respects document boundaries.
    """


    max_len = max_length - 2

    final_input_ids = []
    attention_masks = []

    for i, document in enumerate(documents):

        a = tokenizer(document, return_tensors='np', add_special_tokens=False)
        input_ids = a['input_ids'] #shape = (1, sequence_length)


        end_position = input_ids.shape[1] % max_len
        pad_num = (max_len - end_position) % max_len #stay in mod space to prevent all 0s when perfectly divisible

        input_ids = np.pad(input_ids, [(0,0), (0, pad_num)], mode='constant', constant_values=0)
        input_ids = np.reshape(input_ids, (-1, max_len))
        input_ids = np.pad(input_ids, [(0,0), (1,1)], mode='constant', constant_values=(1,2))

        if pad_num > 0:
            input_ids[-1][end_position+1] = 2
            input_ids[-1][-1] = 0


        attention_mask = (input_ids != 0).astype('int64')


        final_input_ids.extend(np.split(input_ids, input_ids.shape[0]))
        attention_masks.extend(np.split(attention_mask, attention_mask.shape[0]))


    return {
        'input_ids': final_input_ids,
        'attention_mask': attention_masks
    }



def load_data_from_dir(input_dir, tokenizer, max_length=256, chunk=True, download_mode='reuse_dataset_if_exists'):
    """
    Load data directly from text files, tokenize, and chunk.
    :param input_directory:
    :param tokenizer:
    :return:
    """

    num_processes = int(os.environ['SLURM_NTASKS'])

    print('{} processes Available'.format(num_processes))

    if os.path.isdir(input_dir):
        data_files = sorted([os.path.join(input_dir, x) for x in os.listdir(input_dir)])
    else:
        data_files = [input_dir]

    created_datasets = []
    process_function = chunk_and_tokenize_documents if chunk else tokenize_documents
    for i, data_file in enumerate(data_files):
        data = load_dataset('text', data_files=data_file, download_mode=download_mode)
        data = data.filter(lambda ex: len(ex['text']) >= 50)

        data = data.map(
            lambda examples: process_function(examples['text'], tokenizer, max_length), batched=True,
            remove_columns=['text'],
            num_proc=num_processes)

        created_datasets.append(data)
        print('.................Finished with dataset {} of {}'.format(i, len(data_files)))

    created_datasets = [x['train'] for x in created_datasets]
    print('concating:')
    final_dataset = concatenate_datasets(created_datasets)
    return final_dataset

def chunk_dataset_full(dataset, tokenizer, max_length=256, num_special_tokens=2):
    num_processes = int(os.environ['SLURM_NTASKS'])
    print('{} processes Available'.format(num_processes))

    # We want to separate each document with the separator token
    dataset = dataset.map(
        lambda example: {'text': example['text'] + tokenizer.sep_token}, batched=False, num_proc=num_processes
    )

    # Tokenize the examples
    dataset = dataset.map(
        lambda examples: tokenizer(examples['text'], padding=False, truncation=False, add_special_tokens=False,
                                   return_length=True), batched=True, num_proc=num_processes
    )

    # Sort by length and concatenate all input examples together
    dataset = dataset.sort('length')

    dset_tensors = [torch.tensor(d) for d in dataset['input_ids']]
    all_tensors = torch.cat(dset_tensors, 0)

    # Create our new input examples, all but one are sized to max_length - 2
    examples = torch.split(all_tensors, max_length-num_special_tokens, dim=0)

    dataset = Dataset.from_dict({'ids': examples})

    print(dataset)
    print(dataset[0])

    # Encode to proper format
    dataset = dataset.map(
        lambda examples: tokenizer.encode_plus(examples['ids'], add_special_tokens=True, return_length=True),
        batched=False, num_proc=8)

    print(dataset[0])
    print(set(dataset['length']))
    assert all(x <= max_length for x in set(dataset['length']))

    return dataset


def load_data(tokenizer, config, concatenate=True):
    """
    General data loading process. A train set represents a set of data sources which are handled together. Each set will
    be chunked together, and will be treated as a single data source during training.

    :param train_sets: A list of lists denoting sets of data to be used for training.
    :param eval_sets: Not implemented.
    :param tokenizer:
    :param config:
    :return:
    """

    train_datasets = []
    eval_datasets = []

    train_sets = config['train_sets']
    eval_sets = config['eval_sets']

    logging.info('Current train sets: {}'.format(train_sets))
    logging.info('Current eval sets: {}'.format(eval_sets))

    for set in train_sets:
        subset_dataset = []
        for subset in set:
            dataset = load_dataset(config['data_directory'], subset, data_dir=config['data_directory'])
            dataset = dataset['train'] if isinstance(dataset, DatasetDict) else dataset

            subset_dataset.append(dataset)

            if subset in eval_sets:
                raise NotImplementedError

        subset_dataset = concatenate_datasets(subset_dataset)

        if set in config['chunk_subsets']:
            subset_dataset = chunk_dataset_full(subset_dataset, tokenizer)
        else:
            num_processes = int(os.environ['SLURM_NTASKS'])
            subset_dataset = subset_dataset.map(
                lambda examples: tokenizer(examples['text'], truncation=True, max_length=256), batched=True,
                remove_columns=['text'],
                num_proc=num_processes)

        train_datasets.append(subset_dataset)


    if concatenate:
        train_dataset = concatenate_datasets(train_datasets)
    else:
        train_dataset = train_datasets


    eval_dataset = None
    return train_dataset, eval_dataset