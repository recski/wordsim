# wordsim

## Preparations
Building the components requires the installation of build-essential and python-dev packages with `sudo apt-get install build-essential python-dev`.
You must also have [setuptools](https://pypi.python.org/pypi/setuptools) installed for python.

## Dependencies
### 4lang
Install the newest version of [4lang](https://github.com/kornai/4lang). Notes:

* downloadable pre-compiled graphs are sufficient
* you don't have to modify the config files
* set only the `FOURLANGPATH` and `HUNTOOLSBINPATH` environmental variable

### Additional libraries
Install the newest version of:

* [gensim](https://radimrehurek.com/gensim/)
* [glove-python](https://github.com/maciejkula/glove-python)
* [hunmisc](https://github.com/zseder/hunmisc)
* [scikit-learn](http://scikit-learn.org)

## Resources
After preparing the resources you should get the following directory structure:
```
wordsim  
└───resources
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
    └───sim_data
        └───simlex
            └───SimLex-999.txt
```

### Embeddings
* [SENNA](http://ronan.collobert.com/senna/): download and extract the [package](http://ronan.collobert.com/senna/download.html), and execute the `paste hash/words.lst embeddings/embeddings.txt > combined.txt` command. 
* [Huang](http://www.socher.org): download and extract the [ACL2012_wordVectorsTextFile.zip ](http://nlp.stanford.edu/~socherr/ACL2012_wordVectorsTextFile.zip) file, and execute the `paste vocab.txt wordVectors.txt > combined.txt` command.
* [word2vec](https://code.google.com/archive/p/word2vec/): download and extract the [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) file.
* [GloVe](http://nlp.stanford.edu/projects/glove/): download and extract the [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip) file.
* [SP](http://www.cs.huji.ac.il/~roys02/papers/sp_embeddings/sp_embeddings.html/): download and extract the [sp_plus_embeddings_500.dat.gz](http://ie.technion.ac.il/~roiri/data/sp_plus_embeddings_500.dat.gz) file. Insert the `152229 500` line at the beginning of the .dat file with `echo '152229 500' | cat - sp_plus_embeddings_500.dat > sp_plus_embeddings_500.w2v`.
* [Paragram](http://ttic.uchicago.edu/~wieting/): download and extract the [paragram_300_sl999.zip](https://drive.google.com/file/d/0B9w48e1rj-MOck1fRGxaZW1LU2M/view?usp=sharing) file.

### SimLex data
* [SimLex](http://www.cl.cam.ac.uk/~fh295/simlex.html): download and extract the [SimLex-999.zip](http://www.cl.cam.ac.uk/~fh295/SimLex-999.zip) file. 

## Usage
Run `python src/wordsim/regression.py configs/default.cfg` to get regression on features from 6 embeddings (6 features) + wordnet metrics (4 features) + 4lang (2 features). You should get `average correlation: 0.755074732764` as the result.

__NOTE: wordsim requires ca. 15 GB of RAM to load all models__

## Citing
If you use the wordsim system in your experiments, please cite

Gábor Recski, Eszter Iklódi, Katalin Pajkossy, András Kornai: [Measuring semantic similarity of words using concept networks](https://www.aclweb.org/anthology/W16-1622.pdf)
In: Proceedings of the 1st Workshop on Representation Learning for NLP, 2016

```
@InProceedings{Recski:2016c,
  author    = {Recski, G\'{a}bor  and  Ikl\'{o}di, Eszter  and  Pajkossy, Katalin  and  Kornai, Andras},
  title     = {Measuring Semantic Similarity of Words Using Concept Networks},
  booktitle = {Proceedings of the 1st Workshop on Representation Learning for NLP},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics},
  pages     = {193--200}
}
```
