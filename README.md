# wordsim

Requires:
* [gensim](https://radimrehurek.com/gensim/)
* [4lang](https://github.com/kornai/4lang/tree/recski_thesis) (requires newest version, install with `sudo python setup.py install`)
* [hunmisc](https://github.com/zseder/hunmisc) (see above)

### Usage

This'll run regression on features from 3 embeddings + the six 4lang models
`python src/wordsim/regression.py configs/test.cfg`

see config file and code for details, clean-up will come soon

### Original CL experiments
To reproduce 3x3 original experiments from Hill:2015, run:

`ln -s /mnt/store/home/hlt/wordsim resources`

Or get the resources from [here](http://people.mokk.bme.hu/~recski/stuff/resources.tgz).

`python src/wordsim/test.py`

The output will get prettier soon

The mikolov (word2vec) numbers will be based on a larger embedding than the
one used in the paper (they trained a model of their own, I don't want to),
the numbers will be somewhat higher.

### SimLex Test

Wordsim is able to create test wordpair set from SimLex data. 
The path to the [4lang](https://github.com/kornai/4lang/tree/master) folder must be defined as an environment variable with the following key: `FOURLANGPATH`.

To add a new pair to the set run:

`./add_wordpair_to_simlex_test.sh WORD1 WORD2`

To run the wordsim regression (src/regression.py) on the simlex test set run:

`./run_simlex_test.sh`
