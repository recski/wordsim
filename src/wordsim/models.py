import logging
import os
import random

from embedding import type_to_class

class Model(object):

    def featurize(self, w1, w2):
        raise NotImplementedError

class EmbeddingModel(Model):

    def __init__(self, embedding, name):
        self.embedding = embedding
        self.name = name

    def featurize(self, w1, w2):
        sim = self.embedding.get_sim(w1, w2)
        if not sim:
            logging.debug("no sim: {0}".format(w1, w2, sim))
            return {}
        return {self.name: sim}

class DummyModel(Model):

    def featurize(self, w1, w2):
        return dict([(k, random.random()) for k in ('a', 'b', 'c', 'd')])

def get_models(conf):
    models = []
    for e_type in conf.options('embeddings'):
        fn = conf.get('embeddings', e_type)
        path = os.path.join(
            conf.get('global', 'embeddings_path'), e_type, fn)
        e_class = type_to_class[e_type]
        embedding = e_class(path)
        model = EmbeddingModel(embedding, e_type)
        models.append(model)
    return models
