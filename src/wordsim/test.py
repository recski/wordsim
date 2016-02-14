import os

from embedding import Word2VecEmbedding, GloveEmbedding
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
                if e_type == 'word2vec':
                    model = Word2VecEmbedding(path)
                elif e_type == 'glove':
                    model = GloveEmbedding(path)
                else:
                    assert False
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
                print 'testing on data {0} of type {1}'.format(fn, d_type)
                for e_type, models in self.e_models.iteritems():
                    for model, fn in models:
                        print '\ttesting embedding {0} of type {1}'.format(
                            fn, e_type)


class AllTests(WordSimTest):
    sim_data_files = {
        'simlex': ['SimLex-999.txt']}
    embedding_files = {
        'word2vec': ['GoogleNews-vectors-negative300.bin'],
        'glove': ['glove.840B.300d.txt']}


if __name__ == "__main__":
    t = AllTests()
    t.run()
