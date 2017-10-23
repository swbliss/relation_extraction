
import numpy as np
import theano

def init_norm(n_in, n_out):
    W = np.asarray(np.random.randn(n_in, n_out) * 0.1, dtype=theano.config.floatX)
    return W

def load_train_data(datapath, max_l, word_size):
    def fit_length(sent, max_l):
        while len(sent) < max_l:
            sent = np.append(sent, 0)
        return np.asarray(sent[:max_l], dtype='int32')

    decIpt, decTgt = [], []
    f = open(datapath)

    while True:
        entities = f.readline()
        if not entities: break
        num = int(f.readline().strip())  # read away relation
        sent = np.fromstring(f.readline().strip(), dtype="int32", sep=" ")[2:]
        decIpt.append(fit_length(np.insert(sent, 0, word_size - 3), max_l))
        decTgt.append(fit_length(np.append(sent, word_size - 2), max_l))
        for _ in range(num - 1):    # use only first sentence for given entity pair.
            f.readline()

    return np.asarray(decIpt).T, np.asarray(decTgt).T



def get_mask(data):
    mask = (np.not_equal(data, 0)).astype("int32")
    return mask

