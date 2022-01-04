from transformers import XLMRobertaTokenizer, BertTokenizer

def get_xlmr_tokenizer(config):
    tokenizer = XLMRobertaTokenizer.from_pretrained(**config['tokenizer_settings']['init'])
    return tokenizer

def get_mbert_tokenizer(config):
    tokenizer = BertTokenizer.from_pretrained(**config['tokenizer_settings']['init'])
    return tokenizer

def get_tokenizer(config):

    tokenizer_type = config['tokenizer_settings']['tokenizer_type']

    handler = {
        'xlmr_tokenizer' : get_xlmr_tokenizer,
        'mbert_tokenizer': get_mbert_tokenizer
    }

    return handler[tokenizer_type](config)
