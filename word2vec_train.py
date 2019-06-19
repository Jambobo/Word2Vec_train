#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# python word2vec_train/word2vec_train.py data/train.txt word2vec_train/vocabulary.model word2vec_train/vocabulary.vector

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    input_txt, outp1, outp2 = sys.argv[1:4]

    model = Word2Vec(sentences=LineSentence(input_txt),
                     size=100,
                     window=10,
                     min_count=5,
                     sg=1,
                     hs=1,
                     workers=multiprocessing.cpu_count())

    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

