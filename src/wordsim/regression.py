"""many classes and functions taken from github.com/judtacs/semeval/"""

from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression  # nopep8
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from argparse import ArgumentParser
from ConfigParser import ConfigParser
import logging
import cPickle
from featurizer import Featurizer
from numpy import array


def parse_args():
    p = ArgumentParser()
    p.add_argument(
        '-c', '--conf', help='config file', default='None', type=str)
    p.add_argument(
        '-inputs', '--inputs',
        help='input list, for tagging it can be multiple, separated by ,',
        type=str)
    p.add_argument(
        '-outputs', '--outputs', help='output list for tagging separated by ,',
        type=str, default=None)
    p.add_argument(
        '-gold', '--gold',
        help='gold list for training/tagging separated by ,',
        type=str, default=None)
    p.add_argument(
        '-model', help='--model', type=str)
    p.add_argument(
        '-train', help='--train', action='store_true', default=False)
    p.add_argument(
        '-tag', help='--tag', action='store_true', default=False)

    return p.parse_args()


def read_config(args):
    conf = ConfigParser()
    conf.read(args.conf)
    return conf

class RegressionModel:

    def __init__(self, model_name, feat_select_thr=0.0,
                 kernel='poly',
                 degree=2, feats={}, feat_select=True, select_top=50):
        self.model_name = model_name
        self.feat_select_thr = float(feat_select_thr)
        self.feats = feats
        self.kernel = kernel
        self.degree = degree
        self.selector = None
        self.selected_feats = None
        self.manual_select = []
        self.select_top = select_top

    def get_selected_feats(self, support):
        self.selected_feats = {}
        if self.feats != {}:
            reversed_feats = dict([(v, k) for k, v in self.feats.iteritems()])
            for new, old in enumerate(support):
                feat = reversed_feats[old]
                self.selected_feats[feat] = new

    def manual_selection(self, data):
        i2f = dict([(v, k) for k, v in self.feats.iteritems()])
        if self.manual_select != []:
            to_filter_names = []
            for f in self.feats:
                needed = True
                for m in self.manual_select:
                    if m in f:
                        needed = False
                        break
                if needed:
                    to_filter_names.append(f)
        supported = [i for i in i2f if i2f[i] in to_filter_names]
        return data[:, sorted(supported)]

    def preproc_and_train(self, train, train_labels):
        self.manual_select = ['collins', 'wikti', 'twitter']
        self.manual_select = []
        if self.manual_select != []:
            train = self.manual_selection(train)
        self.train(train, train_labels)

    def train(self, data, train_labels):
            if self.model_name == 'sklearn_linear':
                model = linear_model.LinearRegression()
            if self.model_name == 'sklearn_ridge':
                model = linear_model.Ridge()
            if self.model_name == 'sklearn_lasso':
                model = linear_model.Lasso(alpha=0.001)
            if self.model_name == 'sklearn_elastic_net':
                model = linear_model.ElasticNet(alpha=0.001)
            if self.model_name == 'sklearn_kernel_ridge':
                model = kernel_ridge.KernelRidge(
                    alpha=2, kernel=self.kernel, gamma=None,
                    degree=int(self.degree), coef0=1, kernel_params=None)
            if self.model_name == 'sklearn_svr':
                model = svm.SVR(
                    kernel=self.kernel, degree=int(self.degree), coef0=1)
            selection = SelectKBest(k=self.select_top)
            variance = VarianceThreshold(threshold=self.feat_select_thr)
            assert (model, selection, variance)  # silence pyflakes
            print data.shape
            self.pipeline = Pipeline(steps=[
                ('univ_select', SelectKBest(k=10, score_func=f_regression)),
                ('variance', VarianceThreshold(threshold=0.00)),
                ('model', svm.SVR(
                    C=100, cache_size=200, coef0=0.0, epsilon=0.5, gamma=0.1,
                    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001,
                    verbose=False))])

            self.pipeline.fit(data, train_labels)

    def preproc_and_predict(self, data):
        if self.manual_select != []:
            data = self.manual_selection(data)
        print data.shape
        return self.pipeline.predict(data)

class Trainer(object):

    def __init__(self, conf,
                 input_=None, train_labels_fn=None,
                 model='model', model_name='sklearn_svr', feat_select=True,
                 feat_select_thr=0.02, degree=3, kernel='poly', C=100):
        self.model = model
        self.model_name = model_name
        self.feat_select = feat_select
        self.feat_select_thr = feat_select_thr
        self.degree = degree
        self.kernel = kernel
        self.C = C
        self.input_ = input_
        self.train_labels_fn = train_labels_fn
        self.conf = conf

    def get_train_data(self):
        a = Featurizer(self.conf)
        fh = open(self.input_)
        logging.info('featurizing train...')
        sample, labels = a.featurize(fh)
        self.train_labels = array(labels)
        logging.info('Converting table...')
        self.train_data = a.convert_to_table(sample)
        self.feats = a._feat_order

    def featurize_train(self, conf):
        self.get_train_data()
        self.regression_model = RegressionModel(
            model_name=self.model_name, feat_select=self.feat_select,
            feat_select_thr=self.feat_select_thr, degree=self.degree,
            kernel=self.kernel, feats=self.feats)
        self.regression_model.preproc_and_train(
            self.train_data, self.train_labels)

    def dump_model(self):
        fn = self.model
        fh = open(fn, 'w')
        cPickle.dump(self.regression_model, fh)

class Tagger(object):

    def __init__(self, input_=None, model='model',
                 outputs=None, gold=None, conf=None):
        self.input_fns = input_.split(',')
        self.model = model
        if outputs is not None:
            self.output_fns = outputs.split(',')
        else:
            self.output_fns = ['' for i in range(len(self.input_fns))]
        if gold is not None:
            self.gold_fns = gold.split(',')
        else:
            self.gold_fns = ['' for i in range(len(self.input_fns))]
        self.conf = conf

    def get_inputs(self):
        a = Featurizer(self.conf)
        l = []
        for input_ in self.input_fns:
            a = Featurizer(self.conf)
            fh = open(input_)
            logging.info('featurizing input {0}...'.format(input_))
            sample, labels = a.featurize(fh)
            logging.info('Converting table...')
            l.append(a.convert_to_table(sample))
        return l

    def tag(self):
        self.regression_model = cPickle.load(open(self.model))
        self.inputs = self.get_inputs()
        for i, ip in enumerate(self.inputs):
            self.predict_and_eval(
                ip, self.output_fns[i], self.gold_fns[i])

    def predict_and_eval(self, ip, op, gold):

        logging.info('predicting ...'.format(op))
        predicted = self.regression_model.preproc_and_predict(ip)
        if op != '':
            with open(op, 'w') as f:
                f.write('\n'.join(str(i) for i in predicted) + '\n')
        if gold != '':
            with open(gold) as f:
                gold_labels = [float(l.strip()) for l in f]

            logging.info('correlation with {0}:{1}'.format(
                gold, repr(pearsonr(list(predicted), gold_labels))))

def train(args):
        conf = read_config(args)
        a = Trainer(conf, load_feats=args.load_feats, input_=args.inputs,
                    train_labels_fn=args.gold, model=args.model,
                    model_name=conf.get('ml', 'model_name'),
                    feat_select=conf.get('ml', 'feat_select'),
                    feat_select_thr=conf.get('ml', 'feat_select_thr'),
                    degree=conf.get('ml', 'degree'),
                    kernel=conf.get('ml', 'kernel'),
                    C=conf.get('ml', 'C'))
        a.featurize_train(conf)
        a.dump_model()

def tag(args):
    conf = read_config(args)
    a = Tagger(
        load_feats=args.load_feats, input_=args.inputs, model=args.model,
        outputs=args.outputs, gold=args.gold, conf=conf)
    a.tag()

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    if args.train is True:
        train(args)
    elif args.tag is True:
        tag(args)


if __name__ == "__main__":
    main()
