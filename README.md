# wordsim

Requires gensim, 4lang (a setup.py will come soon)

### Usage

This'll run regression on features from 3 embeddings + the six 4lang models
`python src/wordsim/regression.py configs/test.cfg`

see config file and code for details, clean-up will come soon

### Original CL experiments
To reproduce 3x3 original experiments from Hill:2015, run:

`ln -s /mnt/store/home/hlt/wordsim resources`

`python src/wordsim/test.py`

The output will get prettier soon

The mikolov (word2vec) numbers will be based on a larger embedding than the
one used in the paper (they trained a model of their own, I don't want to),
the numbers will be somewhat higher.

