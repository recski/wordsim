"""many classes and funcctions taken from github.com/judtacs/semeval/"""

import os
import random
from numpy import array

from sim_data import SimData
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
            print w1, w2, sim
            return {}
        return {self.name: sim}

class DummyModel(Model):

    def featurize(self, w1, w2):
        return dict([(k, random.random()) for k in ('a', 'b', 'c', 'd')])

class WordPair(object):

    def __init__(self, w1, w2):
        self.pair = (w1, w2)
        self.features = {}

class Featurizer(object):

    def __init__(self, conf):
        self.conf = conf
        self.get_models(self.conf)
        self._feat_order = {}
        self._feat_i = 0

    def get_models(self, conf):
        self.models = []
        for e_type in self.conf.options('embeddings'):
            fn = self.conf.get('embeddings', e_type)
            path = os.path.join(
                self.conf.get('global', 'embeddings_path'), e_type, fn)
            e_class = type_to_class[e_type]
            embedding = e_class(path)
            model = EmbeddingModel(embedding, e_type)
            self.models.append(model)

    def featurize(self):
        sample, labels = [], []
        for data_type in self.conf.options('train_data'):
            fn = self.conf.get('train_data', data_type)
            path = os.path.join(
                self.conf.get('global', 'data_path'), data_type, fn)
            sim_data = SimData.create_from_file(path, data_type)
            for (w1, w2), sim in sim_data.pairs.iteritems():
                pair = WordPair(w1, w2)
                for model in self.models:
                    pair.features.update(model.featurize(w1, w2))
                sample.append(pair)
                labels.append(sim)
        return sample, labels

    def convert_to_table(self, sample):
        table = []
        for s in sample:
            table.append([0] * self._feat_i)
            for feat, sc in s.features.iteritems():
                if not self._feat_order:
                    self._feat_order[feat] = 0
                    self._feat_i = 1
                    table[-1] = [sc]
                elif feat not in self._feat_order:
                    self._feat_order[feat] = self._feat_i
                    self._feat_i += 1
                    table[-1].append(sc)
                else:
                    table[-1][self._feat_order[feat]] = sc
        return array(table)
