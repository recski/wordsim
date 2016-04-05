#!/bin/bash

(cd ${WORDSIMPATH} && python scripts/add_wordpair_to_simlex_test.py -w1 $1 -w2 $2) &&
(cd ${FOURLANGPATH} && python scripts/add_wordpair_to_simlex_test.py -w1 $1 -w2 $2 && python src/fourlang/dict_to_4lang.py conf/simlex_test.cfg) 

