import sys

if __name__ == '__main__':

    lang = sys.argv[1]

    in_file = '/rc_scratch/abeb4417/es_wiki_translations/translations/{0}_es/{0}_es.bpe.hyp'.format(lang)
    out_file = '/projects/abeb4417/americasnli/data/full_translated_wiki/{}.txt'.format(lang)

    with open(in_file, 'r') as f, open(out_file, 'w') as out_f:
        for line in f:
            toks = line.split(' ')
            cleaned = ''.join(toks).replace('▁', ' ')
            out_f.write(cleaned)