import logging
import sys

from keras.layers import Dense, Activation
from keras.models import Sequential
import numpy as np
from scipy.stats import spearmanr

from embedding import GloveEmbedding
from utils import cut


def create_model(input_dim, output_dim):
    model = Sequential()

    # model.add(Dense(output_dim=output_dim, input_dim=input_dim))
    model.add(Dense(output_dim=64, input_dim=input_dim))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=output_dim))
    # model.add(Activation("softmax"))
    model.compile(
        loss='mean_squared_error', optimizer='sgd',
        metrics=["mean_squared_error"])
    return model


def featurize(data, embedding, dim):
    return np.array([featurize_single(e[0], embedding, dim) for e in data])


def featurize_single(data, embedding, dim):
    b1, b2 = data[0].split(), data[1].split()
    vecs = np.array(
        [dim*[0] if v is None else v for v in [
            embedding.get_vec(word) for word in (b1[0], b1[1], b2[0], b2[1])]])
    return vecs.flatten()


def test():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    data = [((f[0], f[1]), float(f[2]))
            for f in [line.strip().split("|||")
                      for line in open(sys.argv[1])]]

    print "sample data:", data[:3]

    train_data, devel_data, test_data = cut(data)

    logging.info('loading model...')
    glove_embedding = GloveEmbedding(sys.argv[2])
    logging.info('done!')
    dim = int(sys.argv[3])
    X_train = featurize(train_data, glove_embedding, dim)

    Y_train = np.array([e[1] for e in train_data])

    logging.info("Input shape: {0}".format(X_train.shape))
    print X_train[:3]
    logging.info("Label shape: {0}".format(Y_train.shape))
    print Y_train[:3]

    input_dim = X_train.shape[1]
    output_dim = 1
    model = create_model(input_dim, output_dim)
    model.fit(X_train, Y_train, nb_epoch=int(sys.argv[4]), batch_size=32)

    X_devel = featurize(devel_data, glove_embedding, dim)
    Y_devel = np.array([e[1] for e in devel_data])

    pred = model.predict_proba(X_devel, batch_size=32)
    corr = spearmanr(pred, Y_devel)
    print "Spearman's R: {0}".format(corr)


if __name__ == "__main__":
    test()
