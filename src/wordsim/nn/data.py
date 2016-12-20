from wordsim.nn.utils import cut


class Dataset(object):
    def __init__(self, raw_data):
        self.data = raw_data

    def vectorize(self, vectorizers):
        pass


def read_ppdb_bigram_data(filename):

    return [((f[0], f[1]), float(f[2]))
            for f in [line.strip().split("|||")
                      for line in open(filename)]]


READERS = {"ppdb_bigram": read_ppdb_bigram_data}


def create_datasets(conf):
    data_fn = conf.get('data', 'file')
    data_type = conf.get('data', 'type')
    reader = READERS[data_type]
    data = reader(data_fn)
    train_data, dev_data, test_data = cut(data)
    train_dataset = Dataset(train_data)
    dev_dataset = Dataset(dev_data)
    test_dataset = Dataset(test_data)
    return train_dataset, dev_dataset, test_dataset
