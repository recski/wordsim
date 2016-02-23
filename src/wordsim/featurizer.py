"""many classes and funcctions taken from github.com/judtacs/semeval/"""

import cPickle
from ConfigParser import ConfigParser
from argparse import ArgumentParser
from numpy import array
import logging

from sim_data import SimData

def parse_args():
    p = ArgumentParser()
    p.add_argument(
        '-c', '--conf', help='config file', default='config', type=str)
    p.add_argument(
        '-i', '--inputs', help='input list, separated by ,', type=str)
    p.add_argument(
        '-o', '--outputs', help='output list, separated by ,', type=str)
    return p.parse_args()


def read_config(args):
    conf = ConfigParser()
    conf.read(args.conf)
    return conf

class Model(object):

    def featurize(self, w1, w2):
        raise NotImplementedError

class DummyModel(Model):

    def featurize(self, w1, w2):
        return {"a": 0.3, "b": 0.5, "c": 0.7}

class WordPair(object):

    def __init__(self):
        self.features = {}

class Featurizer(object):

    def __init__(self, conf):
        self.conf = conf
        self.get_models(self.conf)
        self._feat_order = {}
        self._feat_i = 0

    def get_models(self, conf):
        self.models = [DummyModel()]

    def featurize(self, stream):
        sample, labels = [], []
        for data_type, fns in self.conf.get('data', 'train').iteritems():
            for fn in fns:
                sim_data = SimData.create_from_file(fn, data_type)
                for (w1, w2), sim in sim_data.pairs.iteritems():
                    pair = WordPair()
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

    def dump_data(self, data, labels, fn):
        fh = open(fn, 'w')
        d = {
            'data': data, 'labels': labels, 'config': self.conf,
            'feats': self._feat_order}
        cPickle.dump(d, fh)

    def preproc_data(self, fn, output_fn):
        fh = open(fn)
        sample, labels = self.featurize(fh)
        table = self.convert_to_table(sample)
        self.dump_data(table, labels, output_fn)


def main():
    args = parse_args()
    conf = read_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    a = Featurizer(conf)
    inputs = args.inputs.split(',')
    outputs = args.outputs.split(',')
    for i, f in enumerate(inputs):
        of = outputs[i]
        a.preproc_data(f, of)
    exit()

if __name__ == "__main__":
    main()
