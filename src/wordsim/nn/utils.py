import logging
import random

import numpy as np
from scipy.stats import spearmanr


def cut(data):
    random.seed("lemon and tea")
    random_order = list(range(len(data)))
    random.shuffle(random_order)
    cut_point1 = int(len(data)*0.9)
    cut_point2 = int(len(data)*0.95)
    train_data = [data[i] for i in random_order[:cut_point1]]
    devel_data = [data[i] for i in random_order[cut_point1:cut_point2]]
    test_data = [data[i] for i in random_order[cut_point2:]]
    logging.info(
        "created datasets: {0} in train, {1} in devel, {2} in test".format(
            len(train_data), len(devel_data), len(test_data)))
    return train_data, devel_data, test_data
