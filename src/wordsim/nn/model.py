from keras.layers import Dense, Activation
from keras.models import Sequential


class KerasModel(object):

    def __init__(self, conf, input_dim, output_dim):
        self.conf = conf
        self.model = Sequential()
        self.model.add(Dense(output_dim=64, input_dim=input_dim))
        self.model.add(Activation("relu"))
        self.model.add(Dense(output_dim=output_dim))
        self.model.compile(
            loss='mean_squared_error', optimizer='sgd',
            metrics=["mean_squared_error"])

    def train(self, train_data, epochs, batch_size):
        self.model.fit(
            train_data.data, train_data.labels, nb_epoch=epochs,
            batch_size=batch_size)

    def save(self):
        model_fn = self.conf.get('training', 'model_filename')
        self.model.save(model_fn)
