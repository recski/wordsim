import re
import nltk
from collections import defaultdict

from resources import Resources


class SynsetWrapper(object):

    punct_re = re.compile(r'[\(\)]', re.UNICODE)

    def __init__(self, synset):
        self.synset = synset
        self._lemmas = None
        self._freq = None
        self._hypernyms = None
        self._hyponyms = None
        self._two_link_hypernyms = None
        self._pos = None
        self._similar_tos = None
        self._two_link_similar_tos = None
        self._definition = None

    def __hash__(self):
        return hash(self.synset)

    def definition(self):
        if self._definition is None:
            def_ = self.synset.definition()
            def_ = SynsetWrapper.punct_re.sub(' ', def_)
            self._definition = set(
                [w.strip() for w in def_.split() if (
                    w.strip() and not w.strip() in Resources.stopwords)])
        return self._definition

    def freq(self):
        if self._freq is None:
            self._freq = 0
            for lemma in self.lemmas:
                self._freq += lemma.count()
        return self._freq

    def lemmas(self):
        if self._lemmas is None:
            self._lemmas = []
            for lemma in self.synset.lemmas():
                self._lemmas.append(lemma)
        return self._lemmas

    def hyponyms(self):
        if self._hyponyms is None:
            self._hyponyms = set()
            for h in self.synset.hyponyms():
                self._hyponyms.add(SynsetWrapper(h))
        return self._hyponyms

    def hypernyms(self):
        if self._hypernyms is None:
            self._hypernyms = set()
            for h in self.synset.hypernyms():
                self._hypernyms.add(SynsetWrapper(h))
        return self._hypernyms

    def two_link_hypernyms(self):
        if self._two_link_hypernyms is None:
            self._two_link_hypernyms = set()
            for hyp in self.hypernyms():
                self._two_link_hypernyms |= hyp.hypernyms()
        return self._two_link_hypernyms

    def pos(self):
        if self._pos is None:
            self._pos = self.synset.pos()
        return self._pos

    def similar_tos(self):
        if self._similar_tos is None:
            self._similar_tos = set(
                SynsetWrapper(s) for s in self.synset.similar_tos())
        return self._similar_tos

    def two_link_similar_tos(self):
        if self._two_link_similar_tos is None:
            self._two_link_similar_tos = set()
            for s in self.similar_tos():
                self._two_link_similar_tos |= s.similar_tos()
        return self._two_link_similar_tos


class WordnetCache(object):

    synsets = {}
    synset_to_wrapper = {}
    senses = {}
    boost_cache = {}

    @staticmethod
    def get_significant_synsets(word):
        if word not in WordnetCache.synsets:
            candidates = nltk.corpus.wordnet.synsets(word)
            if len(candidates) == 0:
                WordnetCache.synsets[word] = set()
            else:
                sn = SynsetWrapper(candidates[0])
                WordnetCache.synset_to_wrapper[candidates[0]] = sn
                WordnetCache.synsets[word] = set([sn])
                for c in candidates[1:]:
                    sw = SynsetWrapper(c)
                    WordnetCache.synset_to_wrapper[c] = sw
                    if sw.freq >= 5:
                        WordnetCache.synsets[word].add(sw)
                        continue
                    if sw.lemmas()[0].name() == word and len(sw.lemmas()) < 8:
                        WordnetCache.synsets[word].add(sw)
        return WordnetCache.synsets[word]

    @staticmethod
    def get_senses(word, sense_num=10):
        if word not in WordnetCache.senses:
            WordnetCache.senses[word] = set([word])
            sn = nltk.corpus.wordnet.synsets(word)
            if len(sn) >= sense_num:
                th = len(sn) / 3.0
                for synset in sn:
                    for lemma in synset.lemmas():
                        lsn = nltk.corpus.wordnet.synsets(lemma.name())
                        if len(lsn) <= th:
                            WordnetCache.senses[word].add(
                                lemma.name().replace('_', ' '))
        return WordnetCache.senses[word]

    @staticmethod
    def get_boost(word1, word2):
        if not (word1, word2) in WordnetCache.boost_cache:
            dist = WordnetCache.wordnet_distance(word1, word2)
            WordnetCache.boost_cache[(word1, word2)] = dist
            WordnetCache.boost_cache[(word2, word1)] = dist
        return WordnetCache.boost_cache[(word1, word2)]

    @staticmethod
    def wordnet_distance(word1, word2):
        s1 = WordnetCache.get_significant_synsets(word1)
        s2 = WordnetCache.get_significant_synsets(word2)
        # same synset
        if s1 & s2:
            return 0
        if WordnetCache.is_hypernym(s1, s2):
            return 1
        if WordnetCache.is_two_link_hypernym(s1, s2):
            return 2
        adj1 = set(filter(lambda x: x.pos() == 'a', s1))
        adj2 = set(filter(lambda x: x.pos() == 'a', s2))
        if adj1 and adj2:
            if WordnetCache.is_similar_to(adj1, adj2):
                return 1
            if WordnetCache.is_two_link_similar_to(adj1, adj2):
                return 2
        # TODO decide whether this should go to config or delete it
        # if WordnetCache.is_derivationally_related(s1, s2):
            # return 1
        if WordnetCache.in_glosses(word1, word2, s1, s2):
            return 2
        return None

    @staticmethod
    def is_hypernym(synsets1, synsets2):
        hyps1 = set()
        for s1 in synsets1:
            hyps1 |= set(s1.hypernyms())
        if synsets2 & hyps1:
            return True
        hyps2 = set()
        for s2 in synsets2:
            hyps2 |= set(s2.hypernyms())
        if synsets1 & hyps2:
            return True
        return False

    @staticmethod
    def in_glosses(word1, word2, synsets1, synsets2):
        if (WordnetCache.in_one_glosses(word1, synsets2) or
                WordnetCache.in_one_glosses(word2, synsets1)):
            return True
        return False

    @staticmethod
    def in_one_glosses(word, synsets):
        defs = defaultdict(int)
        for s in synsets:
            for w in s.definition():
                defs[w] += 1
            for h in s.hypernyms():
                for w in h.definition():
                    defs[w] += 1
            for h in s.hyponyms():
                for w in h.definition():
                    defs[w] += 1
        top5 = [i[0] for i in sorted(defs.iteritems(),
                                     key=lambda x: -x[1])[:5]]
        if word in top5:
            return True
        return False

    @staticmethod
    def is_derivationally_related(synsets1, synsets2):
        lemmas1 = set()
        der1 = set()
        for s1 in synsets1:
            lemmas1 |= set(s1.lemmas())
        lemmas2 = set()
        for s2 in synsets2:
            lemmas2 |= set(s2.lemmas())
        for l1 in lemmas1:
            der1 |= set(l1.derivationally_related_forms())
        if der1 & lemmas2:
            return True
        der2 = set()
        for l2 in lemmas2:
            der2 |= set(l2.derivationally_related_forms())
        if der2 & lemmas1:
            return True
        return False

    @staticmethod
    def wn_freq(synset):
        return sum(l.count() for l in synset.lemmas())

    @staticmethod
    def is_two_link_similar_to(adj1, adj2):
        sim1 = set()
        for s in adj1:
            sim1 |= s.two_link_similar_tos()
        if adj2 & sim1:
            return True
        sim2 = set()
        for s in adj2:
            sim2 |= s.two_link_similar_tos()
        if adj1 & sim2:
            return True
        return False

    @staticmethod
    def is_similar_to(adj1, adj2):
        sim1 = set()
        for s in adj1:
            sim1 |= set(s.similar_tos())
        if sim1 & adj2:
            return True
        sim2 = set()
        for s in adj2:
            sim2 |= set(s.similar_tos())
        if sim2 & adj1:
            return True
        return False

    @staticmethod
    def is_two_link_hypernym(synsets1, synsets2):
        hyps1 = set()
        for s in synsets1:
            hyps1 |= s.two_link_hypernyms()
        if synsets2 & hyps1:
            return True
        hyps2 = set()
        for s in synsets2:
            hyps2 |= s.two_link_hypernyms()
        if synsets1 & hyps2:
            return True
        return False
