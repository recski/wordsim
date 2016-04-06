#!/bin/bash

(cd ${FOURLANGPATH} && python setup.py install) &&
echo &&
(cd ${WORDSIMPATH} && python src/wordsim/regression.py configs/simlex_test.cfg)
