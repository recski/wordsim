from ConfigParser import ConfigParser
import logging
import os
import sys

from wordsim.models import get_models
from wordsim.nn.utils import evaluate
from wordsim.nn.data import create_datasets
from wordsim.nn.model import KerasModel


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")
    conf = ConfigParser(os.environ)
    conf.read(sys.argv[1])
    vectorizers = get_models(conf)
    training_data, dev_data, test_data = create_datasets(conf)
    if conf.getboolean('main', 'train'):
        epochs = conf.getint('training', 'epochs')
        batch_size = conf.getint('training', 'batch_size')
        training_data.vectorize(vectorizers)
        input_size, input_dim = training_data.vectors.shape
        model = KerasModel(conf, input_dim, 1)  # output is a score
        model.train(training_data, epochs, batch_size)
        model.save()
    test_data.vectorize(vectorizers)
    evaluate(model, dev_data)


if __name__ == "__main__":
    main()
