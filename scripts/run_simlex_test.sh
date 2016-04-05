#!/bin/bash

(cd ${FOURLANGPATH} && python setup.py install) &&
echo &&
python src/wordsim/regression.py configs/simlex_test.cfg
