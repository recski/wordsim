"""many classes and funcctions taken from github.com/judtacs/semeval/"""

from numpy import array


class WordPair(object):

    def __init__(self, w1, w2):
        self.pair = (w1, w2)
        self.features = {}


class Featurizer(object):

    def __init__(self, conf):
        self.conf = conf
        self._feat_order = {}
        self._feat_i = 0

    def featurize(self, sim_data, models):
        sample, labels = [], []
        for (w1, w2), sim in sim_data.pairs.iteritems():
            pair = WordPair(w1, w2)
            for model in models:
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

    def convert_to_wordpairs(self, sample):
        table = []
        header = ["word1", "word2"]
        for key in sample[0].features:
            header.append(key)
            # split = key.split("_similarity")
            # header.append(split[0])
        header.extend(["4lang", "SimLex", "diff"])
        for s in sample:
            table.append(s.pair)
            # print s.pair[0],' ',s.pair[1],'\n'
            # print s.features,'\n'
        return header, array(table)
