#!/bin/bash

DOWNLOAD_DIR=/rc_scratch/abeb4417/wiki_dumps

date=20210801

for lang in es
do
  LANG_DIR=${DOWNLOAD_DIR}/${lang}
  mkdir ${LANG_DIR} -p
  cd ${LANG_DIR}

  wget -nc https://dumps.wikimedia.org/${lang}wiki/${date}/${lang}wiki-${date}-pages-articles-multistream.xml.bz2 --no-check-certificate
  bzip2 -dk ${lang}wiki-${date}-pages-articles-multistream.xml.bz2

  python -m wikiextractor.WikiExtractor ${lang}wiki-${date}-pages-articles-multistream.xml --processes 12 --no-templates

  mkdir wiki_files -p
  for folder in text/*
  do
    for file in ${folder}/*
    do
      mv ${file} wiki_files/
    done
  done
done
