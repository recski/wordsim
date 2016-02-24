import logging
import os

from scipy.stats import spearmanr

from embedding import type_to_class
from sim_data import SimData

class WordSimTest():
    sim_data_files = ()
    embedding_files = ()

    def __init__(self):
        self.load_sim_data()
        self.load_embeddings()

    def load_embeddings(self):
        self.e_models = {}
        for e_type, file_list in self.embedding_files.iteritems():
            self.e_models[e_type] = []
            for fn in file_list:
                path = os.path.join('resources', 'embeddings', e_type, fn)
                e_class = type_to_class[e_type]
                model = e_class(path)
                self.e_models[e_type].append((model, fn))

    def load_sim_data(self):
        self.sim_datasets = {}
        for d_type, file_list in self.sim_data_files.iteritems():
            self.sim_datasets[d_type] = []
            for fn in file_list:
                path = os.path.join('resources', 'sim_data', d_type, fn)
                data = SimData.create_from_file(path, d_type)
                self.sim_datasets[d_type].append((data, fn))

    def run(self):
        for d_type, datasets in self.sim_datasets.iteritems():
            for data, fn in datasets:
                logging.info(
                    'testing on data {0} of type {1} ({2} pairs)'.format(
                        fn, d_type, len(data.pairs)))
                for e_type, models in self.e_models.iteritems():
                    for model, fn in models:
                        logging.info(
                            '\ttesting embedding {0} of type {1}'.format(
                                fn, e_type))
                        answers, gold_sims, oovs = [], [], 0
                        for (w1, w2), gold in data.pairs.iteritems():
                            sim = model.get_sim(w1, w2)
                            if sim:
                                answers.append(sim)
                                gold_sims.append(gold)
                            else:
                                oovs += 1
                        corr = spearmanr(answers, gold_sims)
                        logging.info('Spearman correlation: {0}'.format(corr))
                        logging.info('pairs skipped (OOVs): {0}'.format(oovs))


class AllTests(WordSimTest):
    sim_data_files = {
        'ws353': ['combined.tab'],
        'men': ['MEN_dataset_natural_form_full'],
        'simlex': ['SimLex-999.txt']}
    embedding_files = {
        'word2vec': ['GoogleNews-vectors-negative300.bin'],
        'huang': ['combined.txt'],
        'senna': ['combined.txt']}
# 'glove': ['glove.840B.300d.txt']}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    t = AllTests()
    t.run()
