"""many classes and functions taken from github.com/judtacs/semeval/"""

from ConfigParser import ConfigParser
import logging
import os
import sys
import time

from sklearn import cross_validation, svm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression  # nopep8
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr
from numpy import array

from featurizer import Featurizer
from sim_data import SimData, type_to_class
from models import get_models


def spearman_scorer(estimator, X, y):
    logging.info('predicting ...')
    predicted = estimator.predict(y)
    return spearmanr(list(predicted), y)


def pearson_scorer(estimator, X, y):
    logging.info('predicting ...')
    predicted = estimator.predict(y)
    return pearsonr(list(predicted), y)


class Regression(object):

    def __init__(self, conf):
        self.conf = conf

    def featurize_data(self, data, models):
        logging.warning('featurizing train...')
        f = Featurizer(self.conf)
        sample, labels = f.featurize(data, models)
        self.labels = array(labels)

        # get word pairs and headers
        self.header, self.words = f.convert_to_wordpairs(sample)

        logging.info('converting table...')
        self.data = f.convert_to_table(sample)
        logging.info('data shape: {0}'.format(self.data.shape))
        logging.info('labels shape: {0}'.format(self.labels.shape))
        self.feats = f._feat_order

    def evaluate(self):
        if self.data.shape[0] < 100:
            return
        self.pipeline = Pipeline(steps=[
            # ('univ_select', SelectKBest(k=10, score_func=f_regression)),
            ('variance', VarianceThreshold(threshold=0.00)),
            ('model', svm.SVR(
                C=100, cache_size=200, coef0=0.0, epsilon=0.5, gamma=0.1,
                kernel='rbf', max_iter=-1, shrinking=True, tol=0.001,
                verbose=False))])

        kf = cross_validation.KFold(len(self.data), n_folds=10)
        X, y = self.data, self.labels
        corrs = []

        iter = 0
        result_str = ''
        test_index_lens = []
        for headerItem in self.header:
            result_str += "{0}\t".format(headerItem)
        result_str += 'iteration\n'

        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.pipeline.fit(X_train, y_train)
            p = self.pipeline.predict(X_test)

            # log result to file
            for i, pred in enumerate(p):
                result_str += "{0}\t{1}\t"\
                    .format(self.words[sum(test_index_lens) + i][0],
                            self.words[sum(test_index_lens) + i][1])
                for feature in X_test[i]:
                    result_str += "{0}\t".format(feature)
                result_str += "{0}\t{1}\t{2}\t{3}\n".format(
                    pred, y_test[i], abs(pred - y_test[i]), iter)
            test_index_lens.append(len(test_index))
            iter += 1

            # corrs.append(pearsonr(p, y_test)[0])
            corrs.append(spearmanr(p, y_test)[0])

        print_results(result_str)
        logging.warning(
            "average correlation: {0}".format(sum(corrs) / len(corrs)))

        # self.pipeline.fit(self.data, self.labels)
        # p = self.pipeline.predict(self.data)
        # print p
        # print pearsonr(p, self.labels)
        # logging.info("running cross-validation...")
        # scores = cross_validation.cross_val_score(
        #     self.pipeline, self.data, self.labels, cv=5, n_jobs=1,
        #     scoring=pearson_scorer)
        # logging.info("scores: {0}".format(scores))


def get_data(conf):
    datasets = {}
    for data_type in conf.options('train_data'):
        if data_type not in type_to_class:
            continue
        fn = conf.get('train_data', data_type)
        path = os.path.join(
            conf.get('global', 'data_path'), data_type, fn)
        datasets[data_type] = SimData.create_from_file(path, data_type)
    return datasets


def print_results(str):
    if not os.path.exists('results'):
        os.makedirs('results')
    time_str = time.strftime("%H%M")
    date_str = time.strftime("%Y%m%d")
    file_str = 'results/res' + date_str + time_str + '.txt'
    file = open(file_str, 'w')
    file.write(str)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    conf = ConfigParser(os.environ)
    conf.read(sys.argv[1])

    logging.warning('loading datasets...')
    datasets = get_data(conf)
    logging.warning('loaded these: {0}'.format(datasets.keys()))
    logging.warning('loading models...')
    models = get_models(conf)
    logging.warning('evaluating...')
    for data_type, data in datasets.iteritems():
        logging.warning('data: {0}'.format(data_type))
        r = Regression(conf)
        r.featurize_data(data, models)
        r.evaluate()

if __name__ == "__main__":
    main()
