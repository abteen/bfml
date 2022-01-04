import os, random

def load_bible(language_set, config):

    loaded_langs = []

    bibles = os.listdir(config['data_settings']['bible_settings']['directory'])

    for lang in language_set:
        for bible in bibles:
            if bible[:3] == lang:
                bible_file = os.path.join(config['data_settings']['bible_settings']['directory'],
                                          bible)
                with open(bible_file, 'r') as f:

                    if config['data_settings']['bible_settings']['subsample'] == 'new_testament':
                        for i, line in enumerate(f, 1):
                            if i >= 26282 and len(line.strip()) > 0:
                                loaded_langs.append(line.strip())

                    elif config['data_settings']['bible_settings']['subsample'] == 'all':
                        for i, line in enumerate(f):
                            if len(line.strip()) > 0:
                                loaded_langs.append(line.strip())


    if config['data_settings']['bible_settings']['shuffle']:
        random.seed(42)
        random.shuffle(loaded_langs)

    return loaded_langs

