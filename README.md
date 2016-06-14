# wordsim

**Required dependencies:**
* [gensim](https://radimrehurek.com/gensim/)
* [4lang](https://github.com/kornai/4lang/tree/recski_thesis) (requires newest version, install with `sudo python setup.py install`)
* [hunmisc](https://github.com/zseder/hunmisc) (see above)

**Required embeddings:**
download these embeddings and put them into the resources/embeddings folder. 
* [SENNA](http://ronan.collobert.com/senna/)
* [Huang](http://www.socher.org)
* [word2vec](https://code.google.com/archive/p/word2vec/)
* [GloVe](https://commoncrawl.org/)
* [SP](http://www.cs.huji.ac.il/ ˜ roys02/papers/sp_embeddings/sp_embeddings.html)
* [Paragram](http://ttic.uchicago.edu/ ˜ wieting/)
 
**SimLex data:**
download the simlex data and extract it into the resources/sim_data/simlex folder.
* [SimLex](http://www.cl.cam.ac.uk/~fh295/SimLex-999.zip)

### Usage
In the config/default.cfg file int the *[lemmatizer]* section, the *hunmorph_path* and the *cache_file* should be set to the appropriate path.

The path to the [4lang](https://github.com/kornai/4lang/tree/master) folder must be defined as an environment variable with the following key: `FOURLANGPATH`.

This'll run regression on features from 6 embeddings (6 features) + wordnet metrics (4 features) + 4lang (2 features)

`python src/wordsim/regression.py configs/default.cfg`

After running it, you should get the following output:

`average correlation: 0.755074732764`

### SimLex Test

Wordsim is able to create test wordpair set from SimLex data. 
The path to the wordsim folder must be defined as an environment variable with the following key: `WORDSIMPATH`.

To add a new pair to the set run:

`scripts/add_wordpair_to_simlex_test.sh WORD1 WORD2`

To run the wordsim regression (src/regression.py) on the simlex test set run:

`scripts/run_simlex_test.sh`
