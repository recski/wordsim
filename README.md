# wordsim

**Required dependencies:**
* [gensim](https://radimrehurek.com/gensim/)
* [4lang](https://github.com/kornai/4lang/tree/recski_thesis) (requires newest version, install with `sudo python setup.py install`)
* [hunmisc](https://github.com/zseder/hunmisc) (see above)

**Required embeddings:**
download these embeddings and place them into the `resources/embeddings` directory with the given subdirectory structures. 
* [SENNA](http://ronan.collobert.com/senna/): download the [senna data](http://ronan.collobert.com/senna/download.html) and execute the `paste hash/words.lst embeddings/embeddings.txt > combined.txt` command. Place the new `combined.txt` file into the `senna` folder. 
* [Huang](http://www.socher.org): download the [ACL2012_wordVectorsTextFile.zip ](http://nlp.stanford.edu/~socherr/ACL2012_wordVectorsTextFile.zip) and execute the `paste vocab.txt wordVectors.txt > combined.txt` command. Place the new `combined.txt` file into the `huang` folder.
* [word2vec](https://code.google.com/archive/p/word2vec/): download the [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and place the `GoogleNews-vectors-negative300.bin` file into the `word2vec` folder. 
* [GloVe](http://nlp.stanford.edu/projects/glove/): download the [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place the `glove.840B.300d.w2v` file into the `glove` folder.
* [SP](http://www.cs.huji.ac.il/~roys02/papers/sp_embeddings/sp_embeddings.html/): download the [sp+ (dim=500)](http://www.cs.huji.ac.il/~roys02/papers/sp_embeddings/sp_plus_embeddings_500.dat.gz). Write the `152229 500` line at the beginning of the `symmp_merged-ppmi-antonym2_10_wn_10000_100_10_random_projection_gaussian_500.dat` file and name the new file: `sp_plus_embeddings_500.w2v`. Put this new file into the `sympat` folder. 
* [Paragram](http://ttic.uchicago.edu/~wieting/): download the [Paragram-SL999](https://drive.google.com/file/d/0B9w48e1rj-MOck1fRGxaZW1LU2M/view?usp=sharing) zip, and place the `paragram_300_sl999.txt` file into the `paragram_300` folder.
 
**SimLex data:**
* [SimLex](http://www.cl.cam.ac.uk/~fh295/simlex.html): download the [SimLex-999](http://www.cl.cam.ac.uk/~fh295/SimLex-999.zip) and place the `SimLex-999.txt` file into the `resources/sim_data/simlex` directory. 

After downloading all of these resources you should have the following directory structure:
```
wordsim  
└───resources
    │
    ├───embeddings
    │   ├───senna
    │   │   └───combined.txt
    │   ├───huang
    │   │   └───combined.txt
    │   ├───word2vec
    │   │   └───GoogleNews-vectors-negative300.bin
    │   ├───glove
    │   │   └───glove.840B.300d.w2v
    │   ├───sympat
    │   │   └───sp_plus_embeddings_500.w2v
    │   └───paragram_300
    │       └───paragram_300_sl999.txt
    │
    └───sim_data
        └───simlex
            └───SimLex-999.txt
```

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
