# wordsim

Requires gensim (a setup.py will come soon)

To reproduce 3x3 original experiments from Hill:2015, run:

`ln -s /mnt/store/home/hlt/wordsim resources`

`python src/wordsim/test.py`

The output will get prettier soon

The mikolov (word2vec) numbers will be based on a larger embedding than the
one used in the paper (they trained a model of their own, I don't want to),
the numbers will be somewhat higher.

