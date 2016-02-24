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

class MachineSimilarity():

    def __init__(self, sim_name, section, cfg):
        from fourlang.lexicon import Lexicon  # nopep8
        from fourlang.similarity import WordSimilarity as FourlangWordSimilarity  # nopep8
        self.fourlang_sim = FourlangWordSimilarity(cfg, section)
        self.sim_name = sim_name
        self.sim_types = cfg.get(section, 'sim_types').split('|')
        for sim_type in self.sim_types:
            if sim_type not in FourlangWordSimilarity.sim_types:
                raise Exception(
                    'unknown 4lang similarity: {0}'.format(sim_type))

    def get_word_sims(self):
        word_sims = {}
        for sim_type in self.sim_types:
            sim_name = "{0}_{1}".format(sim_type, self.sim_name)
            word_sims[sim_name] = self.fourlang_sim.sim_type_to_function(
                sim_type)
        return word_sims

class MachineModel(Model):
    def __init__(self, conf, name):
        self.ms = MachineSimilarity(name, name, conf)
        self.sim_functions = self.ms.get_word_sims()

    def featurize(self, w1, w2):
        return dict(
            [(name, fnc(w1, w2))
             for name, fnc in self.sim_functions.iteritems()])

def get_models(conf):
    models = []
    models.append(MachineModel(conf, 'similarity_machine_longman'))
    for e_type in conf.options('embeddings'):
        fn = conf.get('embeddings', e_type)
        path = os.path.join(
            conf.get('global', 'embeddings_path'), e_type, fn)
        e_class = type_to_class[e_type]
        embedding = e_class(path)
        model = EmbeddingModel(embedding, e_type)
        models.append(model)
    return models
