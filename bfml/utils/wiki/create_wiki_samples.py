import sys, os
if __name__ == '__main__':

    language_dir = os.path.join(sys.argv[1], 'wiki_files')
    wiki_files = os.listdir(language_dir)

    documents = []
    for wiki_file in wiki_files:
        with open(os.path.join(language_dir, wiki_file), 'r') as f:
            for line in f.readlines():
                # Process wiki raw text dumps, preserving documents
                line = line.strip()

                if len(line) > 0:

                    if "<doc id" in line:
                        document = []

                    elif "</doc>" in line:
                        documents.append(document[1:])

                    else:
                        document.append(line)




    output_dir = sys.argv[2]

    with open(os.path.join(output_dir, 'wiki.lbl.es'), 'w') as f:
        for document in documents:
            if len(document) > 0:
                for sentence in document:
                    f.write(sentence + '\n')
                f.write('\n')