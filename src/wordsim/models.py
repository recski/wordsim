from ConfigParser import NoSectionError
import logging
import os
import random

from embedding import type_to_class
from gensim.matutils import unitvec
import numpy as np

from wordnet_cache import WordnetCache as Wordnet


class Model(object):
    def __init__(self):
        self.sim_cache = {}
        self.vec_cache = {}

    def vectorize(self, w1, w2):
        sorted_pair = tuple(sorted((w1, w2)))
        if sorted_pair not in self.vec_cache:
            self.sim_cache[sorted_pair] = dict(self._vectorize(w1, w2))
        return self.sim_cache[sorted_pair]

    def featurize(self, w1, w2):
        sorted_pair = tuple(sorted((w1, w2)))
        if sorted_pair not in self.sim_cache:
            self.sim_cache[sorted_pair] = dict(self._featurize(w1, w2))
        return self.sim_cache[sorted_pair]

    def _vectorize(self, w1, w2):
        raise NotImplementedError

    def _featurize(self, w1, w2):
        raise NotImplementedError


class EmbeddingModel(Model):

    @staticmethod
    def get_unigram_freqs(fn):
        if fn is None:
            return
        unigram_freqs = {}
        for i, line in enumerate(open(fn)):
            try:
                freq, word = line.strip().split()
                unigram_freqs[word] = int(freq)
            except:
                logging.warning('error on freq line {0}: {1}'.format(i, line))

        return unigram_freqs

    def __init__(self, embedding, name, machine_model=None, freq_fn=None):
        super(self.__class__, self).__init__()
        self.embedding = embedding
        self.name = name
        self.machine_model = machine_model
        self.unigram_freqs = EmbeddingModel.get_unigram_freqs(freq_fn)

    def hypsim(self, w1, w2, threshold=5000000000):
        is_freq_1 = self.unigram_freqs.get(w1, 0) >= threshold
        is_freq_2 = self.unigram_freqs.get(w2, 0) >= threshold
        lemma1, lemma2 = [
            self.machine_model.ms.fourlang_sim.lemmatizer.lemmatize(
                word, defined=self.machine_model.ms.fourlang_sim.defined_words,
                stem_first=True)
            for word in (w1, w2)]
        if lemma1 is None:
            logging.warning('OOV: {0}'.format(w1))
            return
        elif lemma2 is None:
            logging.warning('OOV: {0}'.format(w2))
            return

        machine1, machine2 = map(
            lambda m: self.machine_model.ms.fourlang_sim.lexicon.get_machine(
                m, allow_new_oov=False),
            (lemma1, lemma2))
        hypernyms1 = set([h.printname() for h in machine1.hypernyms()])
        hypernyms2 = set([h.printname() for h in machine2.hypernyms()])
        hypernyms2 = set(
            [h for h in hypernyms2 if h not in hypernyms1])  # TODO
        # if (not hypernyms1) or (not hypernyms1):
        #    return

        hyp_vecs_1 = [v for v in (self.embedding.get_vec(h)
                                  for h in hypernyms1) if v is not None]
        hyp_vecs_2 = [v for v in (self.embedding.get_vec(h)
                                  for h in hypernyms2) if v is not None]

        w1_vec = self.embedding.get_vec(w1)
        w2_vec = self.embedding.get_vec(w2)

        all_v_1 = hyp_vecs_1 + [w1_vec] if w1_vec is not None else hyp_vecs_1
        all_v_2 = hyp_vecs_2 + [w2_vec] if w2_vec is not None else hyp_vecs_2
        if all_v_1 == [] or all_v_2 == []:
            print 'foo'
            return

        all_sims = [np.dot(unitvec(v1), unitvec(v2))
                    for v1 in all_v_1 for v2 in all_v_2]
        yield "{0}_hypmax".format(self.name), max(all_sims)
        # ??? This one doesn't seem to help at all (16.05.03)

        if is_freq_1 and is_freq_2:
            hypsim = self.embedding.get_sim(w1, w2)
            if not hypsim:
                hypsim = 0.0
        else:
            # av_vec1 = sum(all_v_1)
            # av_vec2 = sum(all_v_2)
            av_vec1 = sum(hyp_vecs_1)
            av_vec2 = sum(hyp_vecs_2)

            if is_freq_1:
                pair = w1_vec, av_vec2
            elif is_freq_2:
                pair = av_vec1, w2_vec
            else:
                pair = av_vec1, av_vec2

            if isinstance(pair[0], int) or isinstance(pair[1], int):
                # == zero would raise exception for numpy arrays
                return

            hypsim = np.dot(*map(unitvec, pair))

        yield "{0}_hypsim".format(self.name), hypsim

    def get_fourlang_feats(self, w1, w2):
        for name, feat in self.hypsim(w1, w2):
            yield name, feat

        """
        hyp_sim = self.hypsim(w1, w2)
        if hyp_sim:
            yield "{0}_hyps".format(self.name), hyp_sim
        else:
            yield "{0}_hyps".format(self.name), 0
        """

    def _vectorize(self, w1, w2):
        return np.concatenate(map(self.embedding.get_vec, (w1, w2)))

    def _featurize(self, w1, w2):
        sim = self.embedding.get_sim(w1, w2)
        if not sim:
            logging.debug("no sim: {0}".format(w1, w2, sim))
            sim = 0.0

        yield self.name, sim

        if self.machine_model is not None:
            # fl_feats = list(self.get_fourlang_feats(w1, w2))
            # if fl_feats:
            #     print w1, w2, fl_feats
            #     quit()

            for name, value in self.get_fourlang_feats(w1, w2):
                yield name, value


class DummyModel(Model):

    def _featurize(self, w1, w2):
        for k in ('a', 'b', 'c', 'd'):
            yield k, random.random()


class WordnetModel(Model):

    def _featurize(self, w1, w2):
        s1 = Wordnet.get_significant_synsets(w1)
        s2 = Wordnet.get_significant_synsets(w2)
        yield 'wordnet_hyp', float(Wordnet.is_hypernym(s1, s2))
        yield 'wordnet_2-hyp', float(Wordnet.is_two_link_hypernym(s1, s2))
        yield 'wordnet_deriv_rel', float(
            Wordnet.is_derivationally_related(s1, s2))
        yield 'wordnet_in_glosses', float(
            Wordnet.in_glosses(w1, w2, s1, s2))


class MachineSimilarity():

    def __init__(self, sim_name, section, cfg):
        from fourlang.lexicon import Lexicon  # nopep8
        from fourlang.similarity import WordSimilarity as FourlangWordSimilarity  # nopep8
        self.fourlang_sim = FourlangWordSimilarity(cfg, section)
        self.sim_name = sim_name


class MachineModel(Model):
    def __init__(self, conf, name):
        super(self.__class__, self).__init__()
        self.name = name
        self.ms = MachineSimilarity(name, name, conf)

    def _featurize(self, w1, w2):
        features = self.ms.fourlang_sim.phrase_similarities(w1, w2)
        for orig_key, value in features.iteritems():
            key = "{0}_{1}".format(orig_key, self.name)
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


def get_word_models(conf):
    models = {}
    if conf.getboolean('characters', 'enabled'):
        models['char'] = CharacterModel(conf)

    if conf.getboolean('wordnet', 'enabled'):
        models['wordnet'] = WordnetModel()

    for m_type in conf.options('machines'):
        try:
            d = conf.get('machines', m_type)
            model_name = 'similarity_machine_{0}'.format(d)
            conf.options(model_name)
        except NoSectionError:
            continue
        else:
            models[model_name] = MachineModel(conf, model_name)

    if conf.getboolean('embeddings', 'enable_4lang'):
        name = 'similarity_machine_{0}'.format(
            conf.get('embeddings', '4lang_model'))
        if name not in models:  # !!! do not put it there
            fourlang_model_for_embeddings = MachineModel(conf, name)
        else:
            fourlang_model_for_embeddings = models[name]
        freq_file = conf.get('global', 'freq_file')
    else:
        fourlang_model_for_embeddings = None
        freq_file = None

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
            model = EmbeddingModel(
                embedding, e_type, fourlang_model_for_embeddings,
                freq_file)
            models[e_type] = model

    return models

def get_comp_models(conf):
    models = {}
    sim_types = conf.get('composition', 'sim_types').split('|')
    e_type = conf.get('composition', 'embedding')
    try:
        e_class = type_to_class[e_type]
    except KeyError:
        logging.error('embedding type {0} does not exist'.format(e_type))
    else:
        fn = conf.get('embeddings', e_type)
        path = os.path.join(
            conf.get('global', 'embeddings_path'), e_type, fn)
        embedding = e_class(path)
    for c_type in sim_types:
        try:
            c_class = type_to_class[c_type]
        except KeyError:
            continue
        else:
            comp_model = c_class(embedding)
            model = EmbeddingModel(comp_model, c_type)
            models[c_type] = model

    return models


def get_vector_models(conf):
    models = {}
    e_type = conf.get('embedding', 'type')
    try:
        e_class = type_to_class[e_type]
    except KeyError:
        logging.error('embedding type {0} does not exist'.format(e_type))

    fn = conf.get('embedding', 'filename')
    embedding = e_class(fn)
    embedding_model = EmbeddingModel(embedding, e_type)
    models[e_type] = embedding_model
    return models


def get_models(conf):
    if conf.getboolean('similarity', 'word'):
        models = get_word_models(conf)
    elif conf.getboolean('similarity', 'compositional'):
        models = get_comp_models(conf)
    elif conf.getboolean('similarity', 'nn'):
        models = get_vector_models(conf)
    else:
        raise Exception("no similarity type defined!")
    logging.info("model types: {0}".format(models.keys))
    return models.values()
