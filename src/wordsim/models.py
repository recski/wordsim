from ConfigParser import NoSectionError
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


class MachineModel(Model):
    def __init__(self, conf, name):
        super(self.__class__, self).__init__()
        self.ms = MachineSimilarity(name, name, conf)

    def _featurize(self, w1, w2):
        features = self.ms.fourlang_sim.word_similarities(w1, w2)
        for key, value in features.iteritems():
            yield key, value


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
    if conf.getboolean('characters', 'enabled'):
        models.append(CharacterModel(conf))
    for m_type in conf.options('machines'):
        try:
            d = conf.get('machines', m_type)
            model_name = 'similarity_machine_{0}'.format(d)
            conf.options(model_name)
        except NoSectionError:
            continue
        else:
            models.append(MachineModel(conf, model_name))
    for e_type in conf.options('embeddings'):
        try:
            e_class = type_to_class[e_type]
        except KeyError:
            continue
        else:
            fn = conf.get('embeddings', e_type)
            path = os.path.join(
                conf.get('global', 'embeddings_path'), e_type, fn)
            embedding = e_class(path)
            model = EmbeddingModel(embedding, e_type)
            models.append(model)
    return models
