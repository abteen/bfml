import sys, random
from transformers import XLMRobertaTokenizer
if __name__ == '__main__':

    #In this process we use a simple heuristic (char length) to filter out short wiki lines

    lang = sys.argv[1]

    original_file = '/projects/abeb4417/americasnli/data/mlm/{0}/train.{0}'.format(lang)
    translated_file = '/projects/abeb4417/americasnli/data/full_translated_wiki/{}.txt'.format(lang)

    random.seed(42)


    with open(original_file, 'r') as f:
        original_lines = f.readlines()
        original_lines = [sent.strip() for sent in original_lines]


    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    tokenized_ids = tokenizer(original_lines)['input_ids']

    num_subwords = 0
    for sent in tokenized_ids:
        num_subwords += (len(sent) - 2)

    print('Found {} subwords in original text'.format(num_subwords))

    with open(translated_file, 'r') as f:
        translated_lines = f.readlines()
        translated_lines = [sent.strip() for sent in translated_lines if len(sent) > 25]

    random.shuffle(translated_lines)

    selected_translations = []
    subword_total = 0
    upsample_ratio = 1.0
    for sent in translated_lines:
        if subword_total >= (num_subwords * upsample_ratio):
            print('Satisfied upsample ratio')
            break

        selected_translations.append(sent)
        input_ids = tokenizer(sent)['input_ids']
        subword_total += (len(input_ids) - 2)

    print('Selected {} sentences with {} total subwords'.format(len(selected_translations), subword_total))

    final_lines = original_lines + selected_translations
    random.shuffle(final_lines)

    output_file = '/projects/abeb4417/americasnli/data/mlm-translated-{}/train.{}'.format(upsample_ratio, lang)

    with open(output_file, 'w') as f:
        for line in final_lines:
            f.write(line + '\n')