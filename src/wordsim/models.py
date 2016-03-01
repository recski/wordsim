import logging
import os
import random

from embedding import type_to_class


class Model(object):
    def __init__(self):
        self.sim_cache = {}

    def featurize(self, w1, w2):
        sorted_pair = tuple(sorted((w1, w2)))
        if sorted_pair not in self.sim_cache:
            self.sim_cache[sorted_pair] = dict(self._featurize(w1, w2))
        return self.sim_cache[sorted_pair]

    def _featurize(self, w1, w2):
        raise NotImplementedError


class EmbeddingModel(Model):

    def __init__(self, embedding, name):
        super(self.__class__, self).__init__()
        self.embedding = embedding
        self.name = name

    def _featurize(self, w1, w2):
        sim = self.embedding.get_sim(w1, w2)
        if not sim:
            logging.debug("no sim: {0}".format(w1, w2, sim))
            return
        else:
            yield self.name, sim


class DummyModel(Model):

    def _featurize(self, w1, w2):
        for k in ('a', 'b', 'c', 'd'):
            yield k, random.random()


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
        super(self.__class__, self).__init__()
        self.ms = MachineSimilarity(name, name, conf)
        self.sim_functions = self.ms.get_word_sims()

    def _featurize(self, w1, w2):
        for name, fnc in self.sim_functions.iteritems():
            yield name, fnc(w1, w2)


class CharacterModel(Model):
    def __init__(self, conf):
        super(self.__class__, self).__init__()
        self.ns = map(int, conf.get('characters', 'ns').split(','))
        self.types = conf.get('characters', 'types').split(',')
        self.word_cache = {}

    def get_ngrams(self, word, n):
        ngrams = set()
        for i in xrange(len(word) - n + 1):
            ngrams.add(word[i:i + n])
        return ngrams

    def sim_func(self, ng1, ng2, sim_type):
        if sim_type == 'jaccard':
            return float(len(ng1 & ng2)) / len(ng1 | ng2)
        elif sim_type == 'dice':
            return float(2 * len(ng1 & ng2)) / (len(ng1) + len(ng2))
        else:
            assert False

    def _featurize(self, w1, w2):
        for n in self.ns:
            ng1 = self.get_ngrams(w1, n)
            ng2 = self.get_ngrams(w2, n)
            for sim_type in self.types:
                feat_name = "char_{0}_{1}".format(sim_type, n)
                try:
                    feat_value = self.sim_func(ng1, ng2, sim_type)
                except ZeroDivisionError:
                    continue
                yield feat_name, feat_value


def get_models(conf):
    models = []
    models.append(CharacterModel(conf))
    for m_type in conf.options('machines'):
        d = conf.get('machines', m_type)
        models.append(
            MachineModel(conf, 'similarity_machine_{0}'.format(d)))
        models.append(
            MachineModel(conf, 'similarity_machine_{0}_expand'.format(d)))
    for e_type in conf.options('embeddings'):
        fn = conf.get('embeddings', e_type)
        path = os.path.join(
            conf.get('global', 'embeddings_path'), e_type, fn)
        e_class = type_to_class[e_type]
        embedding = e_class(path)
        model = EmbeddingModel(embedding, e_type)
        models.append(model)
    return models
