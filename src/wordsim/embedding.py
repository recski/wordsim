import logging
import sys

from gensim.models import Word2Vec
from glove import Glove
import numpy as np


class Embedding():
    def __init__(self):
        self.E = {}

    def get_sim(self, w1, w2):
        raise NotImplementedError


class Word2VecEmbedding(Embedding):
    @staticmethod
    def load_model(model_fn, model_type):
        logging.info('Loading model: {0}'.format(model_fn))
        if model_type == 'word2vec':
            model = Word2Vec.load_word2vec_format(model_fn, binary=True)
        elif model_type == 'word2vec_txt':
            model = Word2Vec.load_word2vec_format(model_fn, binary=False)
        elif model_type == 'gensim':
            model = Word2Vec.load(model_fn)
        else:
            raise Exception('Unknown model format')
        logging.info('Model loaded: {0}'.format(model_fn))
        return model

    def __init__(self, fn, model_type='word2vec'):
        self.fn = fn
        self.model_type = model_type
        self.model = Word2VecEmbedding.load_model(self.fn, self.model_type)

    def get_sim(self, w1, w2):
        if w1 in self.model and w2 in self.model:
            return self.model.similarity(w1, w2)
        else:
            return None


class GloveEmbedding(Embedding):
    def __init__(self, fn):
        self.fn = fn
        self.model = Glove.load_stanford(fn)

    def get_sim(self, w1, w2):
        if w1 not in self.model.dictionary or w2 not in self.model.dictionary:
            return None
        id1, id2 = map(self.model.dictionary.get, (w1, w2))
        vec1, vec2 = map(lambda i: self.model.word_vectors[i], (id1, id2))
        return (
            np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2))


class TSVEmbedding(Embedding):
    @staticmethod
    def load(fn):
        model = {}
        with open(fn) as f:
            for line in f:
                word, vec_str = line.decode('utf-8').strip().split('\t')
                vec = np.array(map(float, vec_str.split()))
                model[word] = vec
        return model

    def __init__(self, fn):
        self.fn = fn
        self.model = TSVEmbedding.load(fn)

    def get_sim(self, w1, w2):
        vec1, vec2 = map(self.model.get, (w1, w2))
        if vec1 is None or vec2 is None:
            return None
        return (
            np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2))


type_to_class = {
    'word2vec': Word2VecEmbedding,
    'sympat': lambda fn: Word2VecEmbedding(fn, model_type='word2vec_txt'),
    'senna': TSVEmbedding,
    'huang': TSVEmbedding,
    'glove': GloveEmbedding}


test_set = [
    ('king', 'queen'), ('cat', 'dog'), ('cup', 'coffee'), ('coffee', 'tea')]


def test():
    fn, e_type = sys.argv[1:3]
    e_class = type_to_class[e_type]
    model = e_class(fn)
    for w1, w2 in test_set:
        print "{0}\t{1}\t{2}".format(w1, w2, model.get_sim(w1, w2))

if __name__ == "__main__":
    test()
