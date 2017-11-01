import getopt
import os
import pickle
import random
import re
import sys
import time
import dataset
from collections import OrderedDict
from enum import Enum

import lasagne
from lasagne.updates import adagrad
from lasagne.updates import adadelta

import data2cv
from conv_net_classes import *
from dataset import *
from seq2seq import *

class Pretrain(Enum):
    NONE = 0
    SKIPGRAM = 1
    SEQ2SEQ = 2
    DEPSP = 3

def parse_argv(argv):
    opts, args = getopt.getopt(sys.argv[1:], "he:s:u:l:b:w:c:d:i:C:tp:o:L:r:S:m:",
                               ['epoch', 'static', 'hidden_units',
                                'batch_size', 'window', 'active_function',
                                'dimension', 'inputdir', 'curriculum',
                                'test', 'pretrain', 'rnd',])
    epochs = 15
    static = False
    hidden_units_str = '100_10'
    max_l = 80
    batch_size = 50
    window_size = 3
    conv_non_linear = 'tanh'  # active fuction
    dimension = 300
    inputdir = 'data_sem_eval'
    curriculum = 'none'
    for_test = False
    pretrain = 'none'
    rnd = 3435
    optimizer = 'adagrad'
    lr = '0.05'
    context_size = 2
    mode = 0
    for op, value in opts:
        if op == '-e':
            epochs = int(value)
        elif op == '-s':
            static = bool(int(value))
        elif op == '-u':
            hidden_units_str = value
        elif op == '-l':
            max_l = int(value)
        elif op == '-b':
            batch_size = int(value)
        elif op == '-w':
            window_size = int(value)
        elif op == '-a':
            conv_non_linear = value
        elif op == '-d':
            dimension = int(value)
        elif op == '-i':
            inputdir = value
        elif op == '-C':
            curriculum = value
        elif op == '-t':
            for_test = True
        elif op == '-p':
            pretrain = value
        elif op == '-r':
            rnd = int(value)
        elif op == '-o':
            optimizer = value
        elif op == '-L':
            lr = float(value)
        elif op == '-S':
            context_size = int(value)
        elif op == '-m':
            mode = int(value)
        elif op == '-h':
            # TODO (swjung): why we don't control hidden unints
            # usage()
            sys.exit()
    return [epochs, static, hidden_units_str, max_l,
            batch_size, window_size, conv_non_linear, dimension,
            inputdir, curriculum, for_test, pretrain, rnd, optimizer, lr,
            context_size, mode]


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def my_updates(params):
    updates = OrderedDict({})
    for param in params:
        updates[param] = param + 1
    return updates

def L1_norm(params):
    L1 = 0
    ld = 1e-6
    for param in params:
        L1 += ld * T.sum(abs(param))
    return L1

def L2_norm(params):
    L2 = 0
    ld = 1e-6
    for param in params:
        L2 += 0.5 * ld * T.sum(T.sqr(param))
    return L2


def pre_train_get_batch_data(
        indices, nums, sents, poss, eposs, img_h, ctx_size):
    batch_nums = [nums[m] for m in indices]
    batch_sents = [sents[m] for m in indices]
    batch_poss = [poss[m] for m in indices]
    batch_eposs = [eposs[m] for m in indices]

    return pre_train_select_instance(batch_nums,
                                     batch_sents,
                                     batch_poss,
                                     batch_eposs,
                                     img_h,
                                     ctx_size)


def get_batch_data(indices, rels, nums, sents, poss, eposs, test_one, img_h):
    batch_rels = [rels[m][0] for m in indices]  # only consider first relation.
    batch_nums = [nums[m] for m in indices]
    batch_sents = [sents[m] for m in indices]
    batch_poss = [poss[m] for m in indices]
    batch_eposs = [eposs[m] for m in indices]

    # TODO: change this part to remove MIL procedure
    return select_instance(batch_rels,
                           batch_nums,
                           batch_sents,
                           batch_poss,
                           batch_eposs,
                           test_one, img_h)


def pre_train_conv_net(train,
                       test,
                       U,
                       PF1,
                       PF2,
                       seq2seq_data,
                       dep_sp_data,
                       neg_table,
                       filter_hs=3,
                       conv_non_linear="tanh",
                       hidden_units=[100, 51],
                       shuffle_batch=True,
                       epochs=20,
                       sqr_norm_lim=9,
                       lr_decay=0.95,
                       static=False,
                       batch_size=50,
                       img_w=50,
                       pf_dim=5,
                       dropout_rate=[0.5],
                       directory='./',
                       activations_str=[],
                       borrow=True,
                       curriculum="none",
                       for_test=False,
                       rnd=3435,
                       optimizer='adagrad',
                       lr=0.05,
                       pretrain='skipgram',
                       ctx_size=2,
                       mode=0,):
    activations = []
    for act in activations_str:
        dropout_rate.append(0.5)
        if act.lower() == 'tanh':
            activations.append(Tanh)
        elif act.lower() == 'sigmoid':
            activations.append(Sigmoid)

    rng = np.random.RandomState(rnd)
    img_h = len(train[0].sentences[0])  # image height = 101
    filter_w = img_w  # img_w = 50
    # All the sentence are transformed into a picture(2-d matrix). Pad with zeros.
    # The width of the picture equals the dimension of word embedding.
    # The height of the picture equals the number of tokens in the padded sentence.

    feature_maps = hidden_units[0]
    filter_shape = (feature_maps, 1, filter_hs, filter_w + pf_dim * 2)

    x = T.imatrix('x')
    p1 = T.imatrix('pf1')
    p2 = T.imatrix('pf2')
    pool_size = T.imatrix('pos')
    context = T.imatrix('context')
    context_msk = T.imatrix('context_msk')
    di, dm, dt = T.imatrices(3)  # decoderInput, decoderMast, decoderTarget
    if pretrain == Pretrain.SEQ2SEQ:
        neg = T.itensor3('neg')
    else:
        neg = T.imatrix('neg')

    Words = theano.shared(value=U, name="Words")
    PF1W = theano.shared(value=PF1, name="pf1w")
    PF2W = theano.shared(value=PF2, name="pf2w")
    print "--------[1]--------"

    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))])

    zero_vec_tensor = T.vector()
    zero_vec_pf = np.zeros(pf_dim, dtype=theano.config.floatX)
    set_zero_pf1 = theano.function([zero_vec_tensor], updates=[(PF1W, T.set_subtensor(PF1W[0, :], zero_vec_tensor))])
    set_zero_pf2 = theano.function([zero_vec_tensor], updates=[(PF2W, T.set_subtensor(PF2W[0, :], zero_vec_tensor))])
    print "--------[2]--------"

    # The first input layer
    # All the input tokens in a sentence are firstly transformed into vectors by looking up word embeddings.
    input_words = Words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    input_pf1 = PF1W[p1.flatten()].reshape((p1.shape[0], 1, p1.shape[1], pf_dim))
    input_pf2 = PF2W[p2.flatten()].reshape((p2.shape[0], 1, p2.shape[1], pf_dim))

    layer0_input = T.concatenate([input_words, input_pf1, input_pf2], axis=3)

    conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(batch_size, 1, img_h, img_w + pf_dim * 2),
                                    filter_shape=filter_shape, pool_size=pool_size,
                                    non_linear=conv_non_linear, max_window_len=3)
    params = conv_layer.params  # conv parameters
    layer1_input = conv_layer.output.flatten(2)

    assert(pretrain != Pretrain.NONE)
    if pretrain == Pretrain.SEQ2SEQ:
        output_layer = Seq2Seq(words=Words, input=layer1_input, voca_size=U.shape[0],
                               hidden_size=U.shape[1], max_l=max_l, batch_size=batch_size,
                               neg_num=10, lstm_layers_num=1, neg_table=neg_table)
        cost = output_layer.cost(di, dm, dt, neg)
    else:
        output_layer = SkipgramLayer(
            input=layer1_input, words=Words, batch_size=batch_size,img_w=img_w,
            for_test=for_test, ctx_size=ctx_size, max_l=max_l, neg_table=neg_table)
        if pretrain == Pretrain.SKIPGRAM:
            cost = output_layer.cost_skipgram(context_idx=context, neg_idx=neg, mode=mode)
        elif pretrain == Pretrain.DEPSP:
            cost = output_layer.cost_depsp(context_idx=context, context_msk=context_msk, neg_idx=neg, mode=mode)

    if not static:  # if word vectors are allowed to change, add them as model parameters
        params += [Words]
    params += [PF1W]
    params += [PF2W]

    cost += L2_norm(params)
    if optimizer == 'adagrad':
        grad_updates = adagrad(cost, params, learning_rate=lr)
    elif optimizer == 'adadelta':
        grad_updates = adadelta(cost, params, learning_rate=lr, rho=lr_decay)
    else:
        print("optimizer should be adagrad / adadelta")
        raise AssertionError
    # grad_updates = sgd(cost, params, 1e-6)

    print "--------[5]--------"

    # train data split
    # shuffle train dataset and assign to mini batches.
    # if dataset size is not a multiple of mini batches, replicate
    np.random.seed(rnd)

    if len(train) % batch_size > 0:
        extra_data_num = batch_size - len(train) % batch_size
        extra_data = train[:extra_data_num]
        train = np.append(train, extra_data, axis=0)

    # permutation = np.random.permutation(len(train))
    # new_train = train[permutation]

    n_train_batches = train.shape[0] / batch_size

    new_decoder_inputs = None
    new_decoder_targets = None
    new_dep_sp_ctx = None
    if pretrain == Pretrain.SEQ2SEQ:
        decoder_inputs, decoder_targets = seq2seq_data
        if len(decoder_inputs) % batch_size > 0:
            extra_data = decoder_inputs[:, :extra_data_num]
            decoder_inputs = np.append(decoder_inputs, extra_data, axis=1)
            extra_data = decoder_targets[:, :extra_data_num]
            decoder_targets = np.append(decoder_targets, extra_data, axis=1)

        tmp = list(zip(train, decoder_inputs.T, decoder_targets.T))
        np.random.shuffle(tmp)
        new_train, new_decoder_inputs, new_decoder_targets = \
            map(lambda x: np.array(x), zip(*tmp))
        new_decoder_inputs = new_decoder_inputs.T
        new_decoder_targets = new_decoder_targets.T

        valid_decoder_inputs = new_decoder_inputs[:, :n_train_batches // 10 * batch_size]
        valid_decoder_targets = new_decoder_targets[:, :n_train_batches // 10 * batch_size]
        new_decoder_inputs = \
            new_decoder_inputs[:, n_train_batches // 10 * batch_size:]
        new_decoder_targets = \
            new_decoder_targets[:, n_train_batches // 10 * batch_size:]
    elif pretrain == Pretrain.DEPSP:
        if len(dep_sp_data) % batch_size > 0:
            extra_data = dep_sp_data[:extra_data_num, :]
            dep_sp_ctx = np.append(dep_sp_data, extra_data, axis=0)
        tmp = list(zip(train, dep_sp_ctx))
        np.random.shuffle(tmp)
        new_train, new_dep_sp_ctx = map(lambda x: np.array(x), zip(*tmp))
        valid_dep_sp_ctx = new_dep_sp_ctx[:n_train_batches // 10 * batch_size]
        new_dep_sp_ctx = new_dep_sp_ctx[n_train_batches // 10 * batch_size:]
    else:   # Pretrain.SKIPGRAM
        new_train = np.random.permutation(train)

    new_valid = new_train[:n_train_batches // 10 * batch_size]
    new_train = new_train[n_train_batches // 10 * batch_size:]
    n_train_batches = new_train.shape[0]/batch_size
    n_valid_batches = new_valid.shape[0]/batch_size
    # # TODO: change this part to apply curriculum learning
    # if curriculum == "none":
    #     new_train = np.random.permutation(new_train)

    [train_nums, train_sents, train_poss, train_eposs] = pre_train_bags_decompose(new_train)
    if pretrain == Pretrain.SKIPGRAM:
        train_model_batch = theano.function([x, p1, p2, pool_size, context, neg], cost, updates=grad_updates)
    elif pretrain == Pretrain.DEPSP:
        train_model_batch = theano.function([x, p1, p2, pool_size, context, context_msk, neg], cost, updates=grad_updates)
    else:
        train_model_batch = theano.function([x, p1, p2, pool_size, di, dm, dt, neg], cost, updates=grad_updates, on_unused_input='ignore')

    # start training over mini-batches
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print '... pre-training start at  ' + str(now)
    epoch = 0
    if curriculum != "none":
        epochs = 1

    while (epoch < epochs):
        cost_f = open(
            directory + "/pre_trained_data/cost_log_epoch_" + str(epoch), "w")
        print("\n### epoch " + str(epoch) + " ###")
        if curriculum == "none":
            for train_batch_idx in range(n_train_batches):
                if train_batch_idx % 400 == 0:
                    print(" [" + time.asctime(time.localtime(time.time())) +
                          "] " + str(train_batch_idx))

                if train_batch_idx == n_train_batches - 1 \
                        and epoch > epochs - 4:
                    print("#pre_training result saving: " +
                          str(train_batch_idx) + " [" +
                          time.asctime(time.localtime(time.time())) + "]")
                    pickle.dump([conv_layer.W.get_value(), conv_layer.b.get_value(),
                                 Words.get_value(), PF1W.get_value(), PF2W.get_value()],
                                open(directory + "/pre_trained_data/weights_" +
                                     str(epoch) + "_" + str(train_batch_idx) +
                                     ".p", "wb"))
                    print("#pre_training result saving finished: " +
                          str(train_batch_idx) + " [" +
                          time.asctime(time.localtime(time.time())) + "]")

                inst_indices = range(train_batch_idx * batch_size, (train_batch_idx + 1) * batch_size)
                x, p1, p2, pool_size, context_idx = pre_train_get_batch_data(
                    inst_indices, train_nums, train_sents, train_poss,
                    train_eposs, img_h, ctx_size)

                if pretrain == Pretrain.SEQ2SEQ:
                    deIpt = new_decoder_inputs[:, inst_indices]
                    deTgt = new_decoder_targets[:, inst_indices]
                    deMst = utils.get_mask(deIpt)
                    neg_idx = np.asarray(output_layer.table.sample(deIpt.shape[0]*batch_size*10, for_test), dtype='int32')\
                        .reshape((deIpt.shape[0], batch_size, 10))
                    cost = train_model_batch(x, p1, p2, pool_size, deIpt, deMst, deTgt, neg_idx)
                else:
                    if pretrain == Pretrain.DEPSP:
                        context_idx = np.asarray(
                            [new_dep_sp_ctx[m] for m in inst_indices],
                            dtype='int32')
                        context_msk = utils.get_mask(context_idx)
                    neg_num = context_idx.shape[1] * 10
                    neg_idx = np.asarray(output_layer.table.sample(batch_size * neg_num, for_test), dtype='int32') \
                        .reshape((batch_size, neg_num))
                    if pretrain == Pretrain.SKIPGRAM:
                        cost = train_model_batch(x, p1, p2, pool_size, context_idx, neg_idx)
                    else:
                        cost = train_model_batch(x, p1, p2, pool_size, context_idx, context_msk, neg_idx)


                cost_f.write(str((epoch * n_train_batches) + train_batch_idx) +
                             " " + str(cost) + "\n")
                set_zero(zero_vec)
                set_zero_pf1(zero_vec_pf)
                set_zero_pf2(zero_vec_pf)
        epoch += 1
        cost_f.close()

    return [conv_layer.W.get_value(), conv_layer.b.get_value(),
            Words.get_value(), PF1W.get_value(), PF2W.get_value()]


def train_conv_net(train,
                   test,
                   U,
                   PF1,
                   PF2,
                   filter_hs=3,
                   conv_non_linear="tanh",
                   hidden_units=[100, 51],
                   shuffle_batch=True,
                   epochs=25,
                   sqr_norm_lim=9,
                   lr_decay=0.95,
                   static=False,
                   batch_size=50,
                   img_w=50,
                   pf_dim=5,
                   dropout_rate=[0.5],
                   directory='./',
                   activations_str=[],
                   borrow=True,
                   curriculum="none",
                   conv_layer_W=None,
                   conv_layer_b=None,
                   rnd=3435,
                   ):
    # T.config.exception_verbosity='high'
    np.random.seed(rnd)
    rng = np.random.RandomState(rnd)

    activations = []
    for act in activations_str:
        dropout_rate.append(0.5)
        if act.lower() == 'tanh':
            activations.append(Tanh)
        elif act.lower() == 'sigmoid':
            activations.append(Sigmoid)

    img_h = len(train[0].sentences[0])  # image height = 101
    filter_w = img_w  # img_w = 50
    # All the sentence are transformed into a picture(2-d matrix). Pad with zeros.
    # The width of the picture equals the dimension of word embedding.
    # The height of the picture equals the number of tokens in the padded sentence.

    feature_maps = hidden_units[0]
    filter_shape = (feature_maps, 1, filter_hs, filter_w + pf_dim * 2)

    # Train data splitting.
    # Shuffle train dataset and assign to mini batches.
    # If dataset size is not a multiple of mini batches, replicate
    if len(train) % batch_size > 0:
        extra_data_num = batch_size - len(train) % batch_size
        rand_train = np.random.permutation(train)
        extra_data = rand_train[:extra_data_num]
        new_train = np.append(train, extra_data, axis=0)
    else:
        new_train = train
    new_train = np.random.permutation(new_train)
    test = np.random.permutation(test)

    n_train_batches = new_train.shape[0] / batch_size
    valid = new_train[:n_train_batches // 10 * batch_size]
    new_train = new_train[n_train_batches // 10 * batch_size:]
    n_train_batches = new_train.shape[0] / batch_size  # batch number of train data
    n_valid_batches = valid.shape[0] / batch_size  # batch number of valid data

    # TODO: change this part to apply curriculum learning
    if curriculum == "none":
        new_train = np.random.permutation(np.array(new_train))
    elif curriculum == "pos_diff":
        def avg_pos_diff(bag):
            pos_diff_sum = 0.0
            for i in range(bag.num):
                pos_diff_sum += abs(bag.entitiesPos[i][0] - bag.entitiesPos[i][1])
            return pos_diff_sum / bag.num

        new_train = np.array(sorted(new_train, key=lambda bag: avg_pos_diff(bag)))
    elif curriculum == "instance_num":
        new_train = np.array(sorted(new_train, key=lambda bag: bag.num, reverse=True))
    elif curriculum == "sentence_len":
        def original_length(sent):
            temp = sent[:]
            temp.remove(0)
            return len(temp)

        def avg_sent_len(bag):
            length_sum = 0.0
            for sent in bag.sentences:
                length_sum += original_length(sent)
            return length_sum / bag.num

        new_train = np.array(sorted(new_train, key=lambda bag: avg_sent_len(bag)))

    # use gradually more dataset for fast learning experiment
    used_train_batches = []
    full_train_batches = n_train_batches
    train_batches = 1
    used_train_batches.append(train_batches)
    while True:
        train_batches *= 2
        if train_batches < full_train_batches:
            used_train_batches.append(train_batches)
        else:
            break
    used_train_batches.append(full_train_batches)

    for n_batches in used_train_batches[:]:
        np.random.seed(rnd)
        rng = np.random.RandomState(rnd)

        x = T.imatrix('x')
        p1 = T.imatrix('pf1')
        p2 = T.imatrix('pf2')
        pool_size = T.imatrix('pos')
        y = T.ivector('y')

        Words = theano.shared(value=U, name="Words")
        PF1W = theano.shared(value=PF1, name="pf1w")
        PF2W = theano.shared(value=PF2, name="pf2w")

        zero_vec_tensor = T.vector()
        zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
        set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))])

        zero_vec_tensor = T.vector()
        zero_vec_pf = np.zeros(pf_dim, dtype=theano.config.floatX)
        set_zero_pf1 = theano.function([zero_vec_tensor],
                                       updates=[(PF1W, T.set_subtensor(PF1W[0, :], zero_vec_tensor))])
        set_zero_pf2 = theano.function([zero_vec_tensor],
                                       updates=[(PF2W, T.set_subtensor(PF2W[0, :], zero_vec_tensor))])

        # The first input layer
        # All the input tokens in a sentence are firstly transformed into vectors by looking up word embeddings.
        input_words = Words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
        input_pf1 = PF1W[p1.flatten()].reshape((p1.shape[0], 1, p1.shape[1], pf_dim))
        input_pf2 = PF2W[p2.flatten()].reshape((p2.shape[0], 1, p2.shape[1], pf_dim))

        layer0_input = T.concatenate([input_words, input_pf1, input_pf2], axis=3)

        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                        image_shape=(batch_size, 1, img_h, img_w + pf_dim * 2),
                                        filter_shape=filter_shape, pool_size=pool_size,
                                        non_linear=conv_non_linear, max_window_len=3,
                                        W=conv_layer_W, b=conv_layer_b)
        layer1_input = conv_layer.output.flatten(2)

        # the number of hidden unit 0 equals to the features multiple the number of filter (100*1=100)
        # hidden_units[0] = feature_maps*3
        classifier = MLPDropout(rng, input=layer1_input,
                                layer_sizes=[feature_maps * 3, hidden_units[1]],
                                activations=activations,
                                dropout_rates=dropout_rate)
        params = classifier.params  # sofmax parameters
        params += conv_layer.params  # conv parameters

        if not static:  # if word vectors are allowed to change, add them as model parameters
            params += [Words]
        params += [PF1W]
        params += [PF2W]

        [train_rels, train_nums, train_sents, train_poss, train_eposs] = bags_decompose(new_train)
        [valid_rels, valid_nums, valid_sents, valid_poss, valid_eposs] = bags_decompose(valid)
        [test_rels, test_nums, test_sents, test_poss, test_eposs] = bags_decompose(test)

        test_size = 1
        test_input_words = Words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
        test_input_pf1 = PF1W[p1.flatten()].reshape((p1.shape[0], 1, p1.shape[1], pf_dim))
        test_input_pf2 = PF2W[p2.flatten()].reshape((p2.shape[0], 1, p2.shape[1], pf_dim))

        test_layer0_input = T.concatenate([test_input_words, test_input_pf1, test_input_pf2], axis=3)
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size, pool_size)
        test_layer1_input = test_layer0_output.flatten(2)
        p_y_given_x = classifier.predict_p(test_layer1_input)
        test_one = theano.function([x, p1, p2, pool_size], p_y_given_x)

        dropout_cost = classifier.dropout_negative_log_likelihood(y)
        dropout_cost += L2_norm(params)
        grad_updates = adagrad(dropout_cost, params, learning_rate=0.05)
        # grad_updates = adadelta(dropout_cost, params, rho=lr_decay)
        # grad_updates = sgd(dropout_cost, params, 1e-3)
        # grad_updates = my_updates(params)
        train_model_batch = theano.function([x, p1, p2, pool_size, y], dropout_cost, updates=grad_updates, )
        valid_model_batch = theano.function([x, p1, p2, pool_size, y], dropout_cost)

        # start training over mini-batches
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print '... training start at  ' + str(now)

        if curriculum != "none":
            epochs = 1
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        print '\n' + str(now) + '\t batches ' + str(n_batches)
        epoch = 0

        while (epoch < epochs):
            # TODO: change this part to apply curriculum learning
            if curriculum == "none":
                for train_batch_idx in range(n_batches):
                    inst_indices = range(train_batch_idx * batch_size, (train_batch_idx + 1) * batch_size)
                    batch_data = get_batch_data(inst_indices, train_rels, train_nums, train_sents, train_poss,
                                                train_eposs, test_one, img_h)
                    cost = train_model_batch(*batch_data)

                    set_zero(zero_vec)
                    set_zero_pf1(zero_vec_pf)
                    set_zero_pf2(zero_vec_pf)
            # else:
            #     step_size = n_batches / 5 if n_batches / 5 != 0 else 1
            #     for progress in range(n_batches / 5, n_batches + 1, step_size):
            #         prev_valid_cost = sys.maxint
            #         iteration = 0  # number of epochs spent for each curriculum
            #         wait_until_decrease = 0  # epochs spent for waiting decrease of validation cost
            #
            #         while True:
            #             print "progress: " + str(progress / step_size) + ", iteration: ", str(iteration)
            #             for train_batch_idx in range(progress):
            #                 inst_indices = range(train_batch_idx * batch_size, (train_batch_idx + 1) * batch_size)
            #                 batch_data = get_batch_data(inst_indices, train_rels, train_nums, train_sents, train_poss,
            #                                             train_eposs, test_one, img_h)
            #                 train_model_batch(*batch_data)
            #                 set_zero(zero_vec)
            #                 set_zero_pf1(zero_vec_pf)
            #                 set_zero_pf2(zero_vec_pf)
            #
            #             valid_cost = 0
            #             for valid_batch_idx in range(n_valid_batches):
            #                 inst_indices = range(valid_batch_idx * batch_size, (valid_batch_idx + 1) * batch_size)
            #                 batch_data = get_batch_data(inst_indices, valid_rels, valid_nums, valid_sents, valid_poss,
            #                                             valid_eposs, test_one, img_h)
            #                 valid_cost += valid_model_batch(*batch_data)
            #             valid_cost /= float(n_valid_batches)
            #             print("validation cost: " + str(valid_cost))
            #
            #             iteration += 1
            #             if iteration >= 15:  # 15 is the limit of epochs for each curriculum
            #                 break
            #
            #             if valid_cost >= prev_valid_cost:
            #                 wait_until_decrease += 1
            #                 if wait_until_decrease > 4:  # 5 is the limit of waiting time for increase of val cost
            #                     break
            #             else:
            #                 prev_valid_cost = valid_cost
            #                 wait_until_decrease = 0

            # TODO: change this part for {validation | test}
            valid_predict = predict_relation(valid_rels, valid_nums, valid_sents,
                                            valid_poss, valid_eposs, test_one, img_h)
            test_pr = positive_evaluation(valid_predict)

            # test_predict = predict_relation(test_rels, test_nums, test_sents, test_poss, test_eposs, test_one, img_h)
            # test_pr = positive_evaluation(test_predict)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print str(now) + '\t epoch ' + str(epoch) + ' test set PR = [' + str(test_pr[0][-1]) + ' ' + str(
                test_pr[1][-1]) + ']'

            p = test_pr[0][-1]
            r = test_pr[1][-1]

            dir_batch = directory + "/batches_" + str(n_batches) + "/"

            if not os.path.exists(dir_batch):
                os.mkdir(dir_batch)

            save_pr(dir_batch + 'test_pr_' + str(epoch) + '.txt', test_pr)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print str(now) + '\t epoch ' + str(epoch) + ' save PR result...'
            print '\n'
            epoch += 1


def calc_atts(train,
              U,
              PF1,
              PF2,
              filter_hs=3,
              conv_non_linear="tanh",
              hidden_units=[100, 51],
              shuffle_batch=True,
              epochs=25,
              sqr_norm_lim=9,
              lr_decay=0.95,
              static=False,
              batch_size=50,
              img_w=50,
              pf_dim=5,
              dropout_rate=[0.5],
              directory='./',
              inputdir='./',
              activations_str=[],
              borrow=True,
              curriculum="none",
              conv_layer_W=None,
              for_test=False,
              rnd=3435, ):
    theano.config.on_unused_input = 'ignore'

    activations = []
    for act in activations_str:
        dropout_rate.append(0.5)
        if act.lower() == 'tanh':
            activations.append(Tanh)
        elif act.lower() == 'sigmoid':
            activations.append(Sigmoid)

    rng = np.random.RandomState(rnd)
    img_h = len(train[0].sentences[0])  # image height = 101
    filter_w = img_w  # img_w = 50
    # All the sentence are transformed into a picture(2-d matrix). Pad with zeros.
    # The width of the picture equals the dimension of word embedding.
    # The height of the picture equals the number of tokens in the padded sentence.

    feature_maps = hidden_units[0]
    filter_shape = (feature_maps, 1, filter_hs, filter_w + pf_dim * 2)

    x = T.imatrix('x')
    p1 = T.imatrix('pf1')
    p2 = T.imatrix('pf2')
    pool_size = T.imatrix('pos')
    context = T.imatrix('context')  # context words' embedding including entities and words around them

    Words = theano.shared(value=U, name="Words")
    PF1W = theano.shared(value=PF1, name="pf1w")
    PF2W = theano.shared(value=PF2, name="pf2w")
    print "--------[1]--------"

    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))])

    zero_vec_tensor = T.vector()
    zero_vec_pf = np.zeros(pf_dim, dtype=theano.config.floatX)
    set_zero_pf1 = theano.function([zero_vec_tensor], updates=[(PF1W, T.set_subtensor(PF1W[0, :], zero_vec_tensor))])
    set_zero_pf2 = theano.function([zero_vec_tensor], updates=[(PF2W, T.set_subtensor(PF2W[0, :], zero_vec_tensor))])
    print "--------[2]--------"

    # The first input layer
    # All the input tokens in a sentence are firstly transformed into vectors by looking up word embeddings.
    input_words = Words[x.flatten()].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    input_pf1 = PF1W[p1.flatten()].reshape((p1.shape[0], 1, p1.shape[1], pf_dim))
    input_pf2 = PF2W[p2.flatten()].reshape((p2.shape[0], 1, p2.shape[1], pf_dim))

    layer0_input = T.concatenate([input_words, input_pf1, input_pf2], axis=3)

    conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                    image_shape=(batch_size, 1, img_h, img_w + pf_dim * 2),
                                    filter_shape=filter_shape, pool_size=pool_size,
                                    non_linear=conv_non_linear, max_window_len=3, W=conv_layer_W)
    layer1_input = conv_layer.output.flatten(2)

    print "--------[5]--------"

    # train data split
    # shuffle train dataset and assign to mini batches.
    # if dataset size is not a multiple of mini batches, replicate
    np.random.seed(rnd)
    if len(train) % batch_size > 0:
        extra_data_num = batch_size - len(train) % batch_size
        rand_train = np.random.permutation(train)
        extra_data = rand_train[:extra_data_num]
        new_train = np.append(train, extra_data, axis=0)
    else:
        new_train = train
    new_train = np.random.permutation(new_train)

    n_train_batches = new_train.shape[0] / batch_size
    valid = new_train[:n_train_batches // 10 * batch_size]
    new_train = new_train[n_train_batches // 10 * batch_size:]
    n_train_batches = new_train.shape[0] / batch_size  # batch number of train data
    # n_valid_batches = valid.shape[0]/batch_size             # batch number of valid data


    # TODO: change this part to apply curriculum learning
    if curriculum == "none":
        new_train = np.random.permutation(new_train)

    print "--------[3]--------"
    skip_gram = SkipgramLayer(input=layer1_input, words=Words, batch_size=batch_size,
                              img_w=img_w, for_test=for_test, inputdir=inputdir)

    atts = skip_gram.atts(context_idx=context)

    [train_nums, train_sents, train_poss, train_eposs] = pre_train_bags_decompose(new_train)

    # from theano.compile.monitormode import MonitorMode
    # def inspect_inputs(i, node, fn):
    #     print(i, node, "input(s) value(s): ", [input[0] for input in fn.inputs])
    # def inspect_outputs(i, node, fn):
    #     print(" output(s) value(s):", [output[0] for output in fn.outputs])

    calc_atts = theano.function([x, p1, p2, pool_size, context], atts)

    # start training over mini-batches
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print '... pre-training start at  ' + str(now)

    atts_f = open(inputdir + "/pre_trained_data/atts", "w")
    for train_batch_idx in range(n_train_batches):
        if train_batch_idx % 400 == 0:
            print(" [" + time.asctime(time.localtime(time.time())) +
                  "] " + str(train_batch_idx))

        inst_indices = range(train_batch_idx * batch_size, (train_batch_idx + 1) * batch_size)
        x, p1, p2, pool_size, context_idx = pre_train_get_batch_data(inst_indices, train_nums, train_sents, train_poss,
                                                                     train_eposs, img_h)

        att_vals = calc_atts(x, p1, p2, pool_size, context_idx)

        for b in range(batch_size):
            atts_f.write(' '.join(map(lambda x: str(x), train_eposs[inst_indices[b]][0])) + '\n')
            atts_f.write(' '.join(map(lambda x: str(x), train_sents[inst_indices[b]][0])) + '\n')
            atts_f.write(' '.join(map(lambda x: str(x), att_vals[b])) + '\n')

        set_zero(zero_vec)
        set_zero_pf1(zero_vec_pf)
        set_zero_pf2(zero_vec_pf)
    atts_f.close()


def save_model(file, params):
    f = open(file, 'w')
    cPickle.dump(params, f, -1)
    f.close()


def save_pr(file, pr):
    f = open(file, 'w')
    all_pre = pr[0]
    all_rec = pr[1]

    f2 = open(file[:-4] + '_wrong_case.txt', 'w')
    wrong_label = pr[2]
    wrong_answer = pr[3]
    wrong_epos = pr[4]
    wrong_sent = pr[5]

    for i, p in enumerate(all_pre):
        f.write(str(p) + ' ' + str(all_rec[i]) + '\n')

    for i, label in enumerate(wrong_label):
        f2.write(
            ' '.join(list(map(lambda x: str(x), wrong_epos[i]))) + ',\t' +
            ' '.join(list(map(lambda x: str(x), label))) + ',\t' +
            str(wrong_answer[i]) + ',\t' +
            ' '.join(list(map(lambda x: str(x), wrong_sent[i]))) + '\n')

    f.close()
    f2.close()


def positive_evaluation(predict_results):
    predict_y = predict_results[0]
    predict_y_prob = predict_results[1]
    y_given = predict_results[2]
    epos = predict_results[3]
    sent = predict_results[4]

    positive_num = 0
    # find the number of positive examples
    for yi in range(y_given.shape[0]):
        if y_given[yi, 0] > 0:
            positive_num += 1
    # if positive_num == 0:
    #     positive_num = 1
    # sort prob
    index = np.argsort(predict_y_prob)[::-1]

    all_pre = [0]
    all_rec = [0]

    wrong_label = [[0]]
    wrong_answer = [[0]]
    wrong_epos = [[0]]
    wrong_sent = [[0]]
    f_n = 0
    t_p = 0
    f_p = 0
    prev_precision = 100

    def append_wrong_case(labels, py, epos, sent):
        wrong_label.append(labels)
        wrong_answer.append(py)
        wrong_epos.append(epos)
        wrong_sent.append(sent)

    for i in range(y_given.shape[0]):
        labels = y_given[index[i], :]  # key given labels
        py = predict_y[index[i]]  # answer

        if labels[0] == 0:
            # NA bag
            if py > 0:
                f_p += 1
                append_wrong_case(labels, py, epos[index[i]], sent[index[i]])
        else:
            # positive bag
            if py == 0:
                f_n += 1
                append_wrong_case(labels, py, epos[index[i]], sent[index[i]])
            else:
                flag = False
                for j in range(y_given.shape[1]):
                    if j == -1:
                        break
                    if py == labels[j]:
                        flag = True  # true positive
                        break
                if flag:
                    t_p += 1
                else:
                    f_p += 1
                    append_wrong_case(
                        labels, py, epos[index[i]], sent[index[i]])
        if (t_p + f_p) == 0:
            precision = 1
        else:
            precision = float(t_p) / (t_p + f_p)
        recall = float(t_p) / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
            all_pre.append(precision)
            all_rec.append(recall)
            prev_precision = precision
    return [all_pre[1:], all_rec[1:],
            wrong_label[1:], wrong_answer[1:], wrong_epos[1:], wrong_sent[1:]]


def pre_train_select_instance(nums, sents, poss, eposs, img_h, ctx_size):
    numBags = len(sents)
    x = np.zeros((numBags, img_h), dtype='int32')
    p1 = np.zeros((numBags, img_h), dtype='int32')
    p2 = np.zeros((numBags, img_h), dtype='int32')
    pool_size = np.zeros((numBags, 2), dtype='int32')
    context_idx = np.zeros((numBags, 2 + ctx_size*4), dtype='int32')

    for bagIndex, insNum in enumerate(nums):
        maxIns = 0
        x[bagIndex, :] = sents[bagIndex][maxIns]
        p1[bagIndex, :] = poss[bagIndex][maxIns][0]
        p2[bagIndex, :] = poss[bagIndex][maxIns][1]
        pool_size[bagIndex, :] = eposs[bagIndex][maxIns]
        e1_poss = eposs[bagIndex][maxIns][0]
        e2_poss = eposs[bagIndex][maxIns][1]

        for i in range(1 + ctx_size*2):
            cursor1 = e1_poss - ctx_size + i
            cursor2 = e2_poss - ctx_size + i
            context_idx[bagIndex][i] = sents[bagIndex][maxIns][cursor1] \
                if cursor1 in range(len(sents[bagIndex][maxIns])) else 0
            context_idx[bagIndex][i + (1 + ctx_size*2)] = sents[bagIndex][maxIns][cursor2] \
                if cursor2 in range(len(sents[bagIndex][maxIns])) else 0

    return [x, p1, p2, pool_size, context_idx]


def select_instance(rels, nums, sents, poss, eposs, test_one, img_h):
    numBags = len(rels)
    x = np.zeros((numBags, img_h), dtype='int32')
    p1 = np.zeros((numBags, img_h), dtype='int32')
    p2 = np.zeros((numBags, img_h), dtype='int32')
    pool_size = np.zeros((numBags, 2), dtype='int32')
    y = np.asarray(rels, dtype='int32')

    for bagIndex, insNum in enumerate(nums):
        maxIns = 0
        maxP = -1
        if insNum > 1:
            for m in range(insNum):
                insPos = poss[bagIndex][m]
                insX = np.asarray(sents[bagIndex][m], dtype='int32').reshape((1, img_h))
                insPf1 = np.asarray(insPos[0], dtype='int32').reshape((1, img_h))
                insPf2 = np.asarray(insPos[1], dtype='int32').reshape((1, img_h))
                insPool = np.asarray(eposs[bagIndex][m], dtype='int32').reshape((1, 2))
                results = test_one(insX, insPf1, insPf2, insPool)
                p = results[0][y[bagIndex]]
                if p > maxP:
                    maxIns = m
        x[bagIndex, :] = sents[bagIndex][maxIns]
        p1[bagIndex, :] = poss[bagIndex][maxIns][0]
        p2[bagIndex, :] = poss[bagIndex][maxIns][1]
        pool_size[bagIndex, :] = eposs[bagIndex][maxIns]

    return [x, p1, p2, pool_size, y]


def predict_relation(rels, nums, sents, poss, eposs, test_one, img_h):
    numBags = len(rels)
    predict_y = np.zeros((numBags), dtype='int32')
    predict_y_prob = np.zeros((numBags), dtype=theano.config.floatX)
    y = np.asarray(rels, dtype='int32')
    significant_sents = []
    significant_eposs = []

    for bagIndex, insRel in enumerate(rels):
        insNum = nums[bagIndex]
        maxP = -1
        pred_rel_type = 0
        max_pos_p = -1
        max_index = 0
        positive_flag = False
        for m in range(insNum):
            insPos = poss[bagIndex][m]
            insX = np.asarray(sents[bagIndex][m], dtype='int32').reshape((1, img_h))
            insPf1 = np.asarray(insPos[0], dtype='int32').reshape((1, img_h))
            insPf2 = np.asarray(insPos[1], dtype='int32').reshape((1, img_h))
            insPool = np.asarray(eposs[bagIndex][m], dtype='int32').reshape((1, 2))
            results = test_one(insX, insPf1, insPf2, insPool)
            rel_type = results.argmax()
            if positive_flag and rel_type == 0:
                continue
            else:
                # at least one instance is positive
                tmpMax = results.max()
                if rel_type > 0:
                    positive_flag = True
                    if tmpMax > max_pos_p:
                        max_pos_p = tmpMax
                        pred_rel_type = rel_type
                        max_index = m
                else:
                    if tmpMax > maxP:
                        maxP = tmpMax
                        if not positive_flag:
                            max_index = m
        significant_sents.append(sents[bagIndex][max_index])
        significant_eposs.append(eposs[bagIndex][max_index])
        if positive_flag:
            predict_y_prob[bagIndex] = max_pos_p
        else:
            predict_y_prob[bagIndex] = maxP

        predict_y[bagIndex] = pred_rel_type
    return [predict_y, predict_y_prob, y, significant_eposs, significant_sents]


def pre_train_bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_pos = [data_bag.positions for data_bag in data_bags]
    bag_num = [data_bag.num for data_bag in data_bags]
    bag_epos = [data_bag.entitiesPos for data_bag in data_bags]
    return [bag_num, bag_sent, bag_pos, bag_epos]


def bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_pos = [data_bag.positions for data_bag in data_bags]
    bag_num = [data_bag.num for data_bag in data_bags]
    bag_rel = [data_bag.rel for data_bag in data_bags]
    bag_epos = [data_bag.entitiesPos for data_bag in data_bags]
    return [bag_rel, bag_num, bag_sent, bag_pos, bag_epos]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


if __name__ == "__main__":
    epochs, static, hidden_units_str, max_l, batch_size, \
    window_size, conv_non_linear, dimension, inputdir, \
    curriculum, for_test, pretrain, rnd, optimizer, lr,\
    context_size, mode = parse_argv(sys.argv[1:])

    recent_confg = parse_argv(sys.argv[1:])
    recent_f = open('recent_config', 'w')
    for elem in recent_confg:
        recent_f.write(str(elem))
        recent_f.write('\n')
    recent_f.close()

    lasagne.random.set_rng(rnd)
    np.random.seed(rnd)
    random.seed(rnd)

    hu_str = hidden_units_str.split('_')
    hidden_units = [int(hu_str[0])]
    activations = []
    for i in range(1, len(hu_str) - 1, 2):
        hidden_units.append(int(hu_str[i]))
        activations.append(hu_str[i + 1])
    hidden_units.append(int(hu_str[-1]))

    # Read Wv, train, test datasets and save them in the form of python data structure with pickle.
    if not os.path.isfile(inputdir + '/Wv.p') or for_test:
        print '[' + time.asctime(time.localtime()) + '] making wv.p...'
        dataset.wv2pickle(inputdir + '/wv.txt', dimension,
                          inputdir + '/Wv.p', for_test=for_test)
        print '[' + time.asctime(time.localtime()) + '] making wv.p finished.'

    if not os.path.exists('./results_sem_eval/'):
        os.mkdir('./results_sem_eval/')

    # change result dir name as you want
    if pretrain == 'none':
        resultdir = './results_sem_eval/' + str(pretrain) + '_r_' + str(rnd)
    else:
        resultdir = './results_sem_eval/' + str(pretrain) + '_r_' + str(rnd) + '_opt_' + \
                    str(optimizer) + '_lr_' + str(lr).replace('.', '') + '_m_' + str(mode)
        if pretrain == 'skipgram':
            resultdir += '_c_' + str(context_size)

    if not os.path.exists(resultdir):
        os.mkdir(resultdir)
        print 'saving results at ' + resultdir + '...'
    else:
        print(resultdir + " already exists!")
        raise AssertionError

    # resultdir = './' + 'C_' + curriculum + '_e_' + str(epochs) + '_s_' + str(static) + '_u_' + \
    #             hidden_units_str + '_b_' + str(batch_size) + '_w_' + \
    #             str(window_size) + '_c_' + conv_non_linear + '_d_' + \
    #             str(dimension) + '_i_' + inputdir
    # if pretrain != 'none':
    #     resultdir += '_' + pretrain

    print '[' + time.asctime(time.localtime()) + '] load Wv ...'
    Wv = cPickle.load(open(inputdir + '/Wv.p'))
    print '[' + time.asctime(time.localtime()) + '] loading Wv finished.'


    if pretrain == 'none':
        pretrain = Pretrain.NONE
    elif pretrain == 'skipgram':
        pretrain = Pretrain.SKIPGRAM
    elif pretrain == 'seq2seq':
        pretrain = Pretrain.SEQ2SEQ
    elif pretrain == 'depsp':
        pretrain = Pretrain.DEPSP

    if (not os.path.isfile('data_figer/train.p') and pretrain != Pretrain.NONE) or for_test:
        print '[' + time.asctime(time.localtime()) + '] making pre_train.p...'
        dataset.data2pickle('data_figer/train.txt',
                            'data_figer/train.p', for_test, word_size=Wv.shape[0])
        print '[' + time.asctime(time.localtime()) + '] making pre_train.p finished.'
    if (not os.path.isfile('data_neg_sampling/neg_table_' + inputdir.replace('/', '_') + '.p') and pretrain != Pretrain.NONE) or for_test:
        print '[' + time.asctime(time.localtime()) + '] making neg_table_' + inputdir.replace('/', '_') + '.p...'
        table = negsampling.UnigramTable(dict_dir=inputdir)
        f = open('data_neg_sampling/neg_table_' + inputdir.replace('/', '_') + '.p', 'w')
        cPickle.dump(table, f, -1)
        f.close()
        print '[' + time.asctime(time.localtime()) + '] making neg_table.p finished.'
    if (not os.path.isfile('data_figer/seq2seq_train.p') and pretrain == Pretrain.SEQ2SEQ) or for_test:
        print '[' + time.asctime(time.localtime()) + '] making seq2seq_train.p...'
        dataset.data2pickle_seq2seq('data_figer/train.txt',
                                    'data_figer/seq2seq_train.p', max_l, Wv.shape[0])
        print '[' + time.asctime(time.localtime()) + '] making seq2seq_train.p finished.'
    if (not os.path.isfile('data_figer/dep_sp.p') and pretrain == Pretrain.DEPSP) or for_test:
        print '[' + time.asctime(time.localtime()) + '] making dep_sp.p...'
        dataset.data2pickle('data_figer/dep_sp.txt', 'data_figer/dep_sp.p',
                            for_test, for_dep_sp=True, max_l=max_l)
        print '[' + time.asctime(time.localtime()) + '] making dep_sp.p finished.'
    if not os.path.isfile(inputdir + '/test.p') or for_test:
        print '[' + time.asctime(time.localtime()) + '] making test.p...'
        dataset.data2pickle(inputdir + '/test.txt',
                            inputdir + '/test.p', for_test, word_size=Wv.shape[0])
        print '[' + time.asctime(time.localtime()) + '] making test.p finished.'
    if not os.path.isfile(inputdir + '/train.p') or for_test:
        print '[' + time.asctime(time.localtime()) + '] making train.p...'
        dataset.data2pickle(inputdir + '/train.txt',
                            inputdir + '/train.p', for_test, word_size=Wv.shape[0])
        print '[' + time.asctime(time.localtime()) + '] making train.p finished.'

    print '[' + time.asctime(time.localtime()) + \
          '] load pretrain/test/train/negTable data...'

    pretrainData = None
    seq2seqData = None
    depspData = None
    negTable = None
    if pretrain != Pretrain.NONE:
        pretrainData = cPickle.load(open('data_figer/train.p'))
        negTable = cPickle.load(open('data_neg_sampling/neg_table_' + inputdir.replace('/', '_') + '.p'))
    if pretrain == Pretrain.SEQ2SEQ:
        seq2seqData = cPickle.load(open('data_figer/seq2seq_train.p'))
    if pretrain == Pretrain.DEPSP:
        depspData = cPickle.load(open('data_figer/dep_sp.p'))
    testData = cPickle.load(open(inputdir + '/test.p'))
    trainData = cPickle.load(open(inputdir + '/train.p'))
    print '[' + time.asctime(time.localtime()) + \
          '] loading pretrain/test/train/negTable data finished.'
    # testData = testData[1:5]
    # trainData = trainData[1:15]
    # tmp = inputdir.split('_')

    if pretrain != Pretrain.NONE:
        pretrainData = data2cv.make_idx_data_cv(pretrainData, window_size, max_l)
    test = data2cv.make_idx_data_cv(testData, window_size, max_l)
    train = data2cv.make_idx_data_cv(trainData, window_size, max_l)

    # Construct position embedding matrices.
    rng = np.random.RandomState(rnd)
    PF1 = np.asarray(rng.uniform(
        low=-1, high=1, size=[2 * max_l - 1, 5]), dtype=theano.config.floatX)
    PF2 = np.asarray(rng.uniform(
        low=-1, high=1, size=[2 * max_l - 1, 5]), dtype=theano.config.floatX)
    conv_layer_W = None
    conv_layer_b = None

    import sys
    sys.setrecursionlimit(10000)

    if pretrain != Pretrain.NONE:
        if not os.path.exists(resultdir + "/pre_trained_data/"):
            os.mkdir(resultdir + "/pre_trained_data/")
        data = os.listdir(resultdir + "/pre_trained_data")
        if len(data) != 0:
            data.sort(key=natural_keys)
            print '[' + time.asctime(time.localtime()) + \
                  "] Loading pre-trained weights... " + str(data[-1])
            [conv_layer_W, conv_layer_b, Wv, PF1, PF2] = pickle.load(
                open(resultdir + "/pre_trained_data/" + str(data[-1]), "rb"))
            print '[' + time.asctime(time.localtime()) + \
                  "] Loading pre-trained weights finished! " + str(data[-1])
        else:
            print '[' + time.asctime(time.localtime()) + "] pre_train_conv_net started..."
            conv_layer_W, conv_layer_b, Wv, PF1, PF2 = pre_train_conv_net(
                            train=pretrainData,
                            test=test,
                            U=Wv,
                            PF1=PF1,
                            PF2=PF2,
                            seq2seq_data=seq2seqData,
                            dep_sp_data=depspData,
                            neg_table=negTable,
                            filter_hs=window_size,
                            conv_non_linear=conv_non_linear,
                            hidden_units=hidden_units,
                            epochs=epochs,
                            static=static,
                            batch_size=batch_size,
                            img_w=dimension,
                            directory=resultdir,
                            activations_str=activations,
                            curriculum=curriculum,
                            for_test=for_test,
                            rnd=rnd,
                            optimizer=optimizer,
                            lr=lr,
                            pretrain=pretrain,
                            ctx_size=context_size,
                            mode=mode,
                            )

    '''
    print '[' + time.asctime(time.localtime()) + "] calc_atts started..."
    calc_atts(pretrain,
            Wv,
            PF1,
            PF2,
            filter_hs=window_size,
            conv_non_linear=conv_non_linear,
            hidden_units=hidden_units,
            shuffle_batch=True,
            epochs=epochs,
            static=static,
            directory=resultdir,
            inputdir=inputdir,
            batch_size=batch_size,
            img_w=dimension,
            curriculum=curriculum,
            conv_layer_W=conv_layer_W,
            rnd=rnd)
    '''

    print '[' + time.asctime(time.localtime()) + "] train_conv_net started..."
    train_conv_net(train,
                   test,
                   Wv,
                   PF1,
                   PF2,
                   filter_hs=window_size,
                   conv_non_linear=conv_non_linear,
                   hidden_units=hidden_units,
                   activations_str=activations,
                   shuffle_batch=True,
                   epochs=15,
                   directory=resultdir,
                   batch_size=batch_size,
                   img_w=dimension,
                   curriculum=curriculum,
                   conv_layer_W=conv_layer_W,
                   conv_layer_b=conv_layer_b,
                   rnd=rnd
                   )
