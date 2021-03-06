import numpy
import theano
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.nnet import sigmoid
# from theano.tensor.nnet import conv
from theano.tensor.signal import pool
from nonlinear import *
from theano.ifelse import ifelse
import negsampling


class HiddenLayer(object):
    """
    Class for HiddenLayer
    """

    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            if activation.func_name == "ReLU":
                W_values = numpy.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)),
                                         dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
                                                     high=numpy.sqrt(6. / (n_in + n_out)),
                                                     size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))

        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
"""
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
            activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLPDropout(object):
    """A multilayer perceptron with dropout"""

    def __init__(self, rng, input, layer_sizes, dropout_rates, activations,
                 use_bias=True, W=None, b=None):

        # rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        next_layer_input = input
        # first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                                                    input=next_dropout_layer_input,
                                                    activation=activations[layer_counter],
                                                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                                                    dropout_rate=dropout_rates[layer_counter])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the parameters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                                     input=next_layer_input,
                                     activation=activations[layer_counter],
                                     # scale the weight matrix W with (1-p)
                                     W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                                     b=next_dropout_layer.b,
                                     n_in=n_in, n_out=n_out,
                                     use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            # first_layer = False
            layer_counter += 1

        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
            input=next_dropout_layer_input,
            n_in=n_in, n_out=n_out, W=W, b=b)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        self.y_pred = self.layers[-1].y_pred
        # zdj added
        self.p_y_given_x = self.layers[-1].p_y_given_x
        # Grab all the parameters together.
        self.params = [param for layer in self.dropout_layers for param in layer.params]

    def predict(self, new_data):
        next_layer_input = new_data
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                next_layer_input = self.activations[i](T.dot(next_layer_input, layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred

    def predict_p(self, new_data):
        next_layer_input = new_data
        for i, layer in enumerate(self.layers):
            print i, layer
            if i < len(self.layers) - 1:
                next_layer_input = self.activations[i](T.dot(next_layer_input, layer.W) + layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) + layer.b)
        return p_y_given_x


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the
    architecture (one minibatch)

    :type n_in: int
    :param n_in: number of input units, the dimension of the space in
    which the datapoints lie

    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
    which the labels lie

    """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

    .. math::

    \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
    \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
    \ell (\theta=\{W,b\}, \mathcal{D})

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label

    Note: we use the mean instead of the sum so that
    the learning rate is less dependent on the batch size
    """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
    zero one loss over the size of the minibatch

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
    correct label
    """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, pool_size,
                 non_linear="tanh", max_window_len=3, W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        # self.poolsize = pool_size
        self.non_linear = non_linear
        self.max_window_len = max_window_len
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / image_shape[2])
        # initialize weights with random weights
        if not isinstance(W, type(None)):
            # avg = numpy.average(W)
            # std = numpy.std(W)
            # W = (W - avg) / std
            # W /= numpy.abs(numpy.max(W)) * 100
            self.W = theano.shared(numpy.asarray(W, dtype=theano.config.floatX), borrow=True, name="W_conv")
        elif self.non_linear == "none" or self.non_linear == "relu":
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                                                 dtype=theano.config.floatX), borrow=True, name="W_conv")
        else:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                 dtype=theano.config.floatX), borrow=True, name="W_conv")

        if not isinstance(b, type(None)):
            self.b = theano.shared(numpy.asarray(b, dtype=theano.config.floatX), borrow=True, name="b_conv")
        else:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name="b_conv")

        # convolve input feature maps with filters
        conv_out = conv2d(input=input, filters=self.W, filter_shape=self.filter_shape, input_shape=self.image_shape)
        # conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        pool_list = []
        if self.non_linear == "tanh" or self.non_linear == "relu" or self.non_linear == "sigmoid":
            if self.non_linear == "tanh":
                conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            elif self.non_linear == "relu":
                conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            else:
                conv_out_tanh = Sigmoid(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

            # pad_len = int(self.max_window_len/2)
            # right_pad_len = int(filter_shape[2]/2)
            # index_shift = pad_len-right_pad_len
            index_shift = int(filter_shape[2] / 2)
            for i in xrange(image_shape[0]):
                # partition sentence via pool size
                e1pos = pool_size[i, 0] + index_shift
                e2pos = pool_size[i, 1] + index_shift
                # if T.gt(e1pos, 0):
                #     p1 = conv_out_tanh[i, :, :e1pos, :]
                # else:
                # p1 = conv_out_tanh[i, :, 0, :].dimshuffle(0, 1, 'x', 2)
                # p1 = conv_out_tanh[i, :, 0:1, :]
                p1 = conv_out_tanh[i, :, :e1pos, :]
                p2 = conv_out_tanh[i, :, e1pos:e2pos, :]
                p3 = conv_out_tanh[i, :, e2pos:, :]

                # p1 = conv_out_tanh[i, :, 1:5, :]
                # p2 = conv_out_tanh[i, :, 1:5, :]
                # p3 = conv_out_tanh[i, :, 1:5, :]
                p1_pool_out = T.max(p1, axis=1)
                p2_pool_out = T.max(p2, axis=1)
                p3_pool_out = T.max(p3, axis=1)
                temp = T.concatenate([p1_pool_out, p2_pool_out, p3_pool_out], axis=1)
                pool_list.append(temp.dimshuffle('x', 0, 1))
        else:
            pass
        self.output = T.concatenate(pool_list, axis=0)
        self.params = [self.W, self.b]

    def predict(self, new_data, batch_size, pool_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, input_shape=img_shape)
        # conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        pool_list = []
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            # pad_len = int(self.max_window_len/2)
            # right_pad_len = int(self.filter_shape[2]/2)
            # index_shift = pad_len-right_pad_len
            index_shift = int(self.filter_shape[2] / 2)
            for i in xrange(batch_size):
                # partition sentence via pool size
                e1pos = pool_size[i, 0] + index_shift
                e2pos = pool_size[i, 1] + index_shift
                # if T.gt(e1pos, 0):
                #     p1 = conv_out_tanh[i, :, :e1pos, :]
                # else:
                #     p1 = conv_out_tanh[i, :, 0, :]
                p1 = conv_out_tanh[i, :, :e1pos, :]
                p2 = conv_out_tanh[i, :, e1pos:e2pos, :]
                p3 = conv_out_tanh[i, :, e2pos:, :]
                p1_pool_out = T.max(p1, axis=1)
                p2_pool_out = T.max(p2, axis=1)
                p3_pool_out = T.max(p3, axis=1)
                temp = T.concatenate([p1_pool_out, p2_pool_out, p3_pool_out], axis=1)
                pool_list.append(temp.dimshuffle('x', 0, 1))
        else:
            pass
        output = T.concatenate(pool_list, axis=0)
        return output


class SkipgramLayer(object):
    def __init__(self, input, words, batch_size, img_w,
                 for_test, ctx_size, max_l, neg_table):
        self.input = input
        self.words = words
        self.batch_size = batch_size
        self.img_w = img_w
        self.table = neg_table
        self.ctx_size = ctx_size
        self.max_l = max_l
        self.for_test = for_test

    def atts(self, context_idx):
        atts = T.zeros((self.batch_size, 2 + self.ctx_size*4), dtype=theano.config.floatX)
        context_wvs = self.words[context_idx.flatten()].reshape(
            (self.batch_size, 2 + self.ctx_size*4, self.img_w)).transpose((0, 2, 1))

        for b in range(self.batch_size):
            total_cos_sim = 0
            total_cos_sim2 = 0
            for c in range(2 + self.ctx_size*4):
                if c == self.ctx_size or c == self.ctx_size*3 + 1:  # e1
                    continue
                elif c < self.ctx_size*2  + 1:
                    cos_sim = T.exp(
                        T.dot(self.input[b, :], context_wvs[b, :, c]))
                    total_cos_sim += cos_sim
                    atts = T.set_subtensor(atts[b, c], cos_sim)
                else:
                    cos_sim = T.exp(
                        T.dot(self.input[b, :], context_wvs[b, :, c]))
                    total_cos_sim2 += cos_sim
                    atts = T.set_subtensor(atts[b, c], cos_sim)

            for c in range(2 + self.ctx_size*4):
                if c == self.ctx_size or c == self.ctx_size*3 + 1:  # e1
                    continue
                elif c < self.ctx_size*2  + 1:
                    atts = T.set_subtensor(atts[b, c], atts[b, c] / total_cos_sim)
                else:
                    atts = T.set_subtensor(atts[b, c], atts[b, c] / total_cos_sim2)
        '''
        for b in range(self.batch_size):
            total_cos_sim = 0
            for c in range(2 + self.ctx_size*4):
                # if not (c == 2 or c == 7):
                cos_sim = T.exp(
                    T.dot(self.input[b, :], context_wvs[b, :, c]))
                total_cos_sim += cos_sim
                atts = T.set_subtensor(atts[b, c], cos_sim)
            atts = T.set_subtensor(atts[b, :], atts[b, :] / total_cos_sim)
        '''
        return atts

    def cost_skipgram(self, context_idx, neg_idx, mode):
        context_wvs = self.words[context_idx.flatten()].reshape(
            (self.batch_size, 2 + self.ctx_size*4, self.img_w,)).transpose((0, 2, 1))
        neg_wvs = self.words[neg_idx.flatten()].reshape(
            (self.batch_size, (2 + self.ctx_size*4)*10, self.img_w)).transpose((0, 2, 1))

        new_context_wvs = context_wvs
        new_neg_wvs = neg_wvs

        # mode 0: e1, e2, c1, c2, ..., c8
        # mode 1: e1, e2, context_words_att
        # mode 2: e1, e2, context_words_att(e1), context_words_att(e2)
        # mode 3: e1_context_words_att, e2_context_words_att
        # mode 4: context_words_att(all)
        if mode != 0:
            assert(self.ctx_size != 0)

        if mode == 1:
            new_context_wvs = T.zeros_like(context_wvs[:, :, :3], dtype=theano.config.floatX)
            for b in range(self.batch_size):
                c_wvs = numpy.zeros((self.img_w,), dtype=theano.config.floatX)
                total_cos_sim = 0
                for c in range(2 + self.ctx_size*4):
                    if c == self.ctx_size:  # e1
                        new_context_wvs = \
                            T.set_subtensor(new_context_wvs[b, :, 0], context_wvs[b, :, c])
                    elif c == self.ctx_size*3 + 1:  # e2
                        new_context_wvs = \
                            T.set_subtensor(new_context_wvs[b, :, 1], context_wvs[b, :, c])
                    else:
                        cos_sim = T.exp(T.dot(self.input[b], context_wvs[b, :, c]))
                        total_cos_sim += cos_sim
                        c_wvs += cos_sim * context_wvs[b, :, c]
                new_context_wvs = T.set_subtensor(new_context_wvs[b, :, 2], c_wvs / total_cos_sim)
            new_neg_wvs = neg_wvs[:, :, :3 * 10]
        elif mode == 2:
            new_context_wvs = T.zeros_like(context_wvs[:, :, :4], dtype=theano.config.floatX)
            for b in range(self.batch_size):
                c_wvs = numpy.zeros((self.img_w,), dtype=theano.config.floatX)
                total_cos_sim = 0
                c_wvs2 = numpy.zeros((self.img_w,), dtype=theano.config.floatX)
                total_cos_sim2 = 0
                for c in range(2 + self.ctx_size*4):
                    if c == self.ctx_size:  # e1
                        new_context_wvs = \
                            T.set_subtensor(new_context_wvs[b, :, 0], context_wvs[b, :, c])
                    elif c == self.ctx_size*3 + 1:  # e2
                        new_context_wvs = \
                            T.set_subtensor(new_context_wvs[b, :, 1], context_wvs[b, :, c])
                    elif c < self.ctx_size*2 + 1:
                        cos_sim = T.exp(T.dot(self.input[b], context_wvs[b, :, c]))
                        total_cos_sim += cos_sim
                        c_wvs += cos_sim * context_wvs[b, :, c]
                    else:
                        cos_sim = T.exp(T.dot(self.input[b], context_wvs[b, :, c]))
                        total_cos_sim2 += cos_sim
                        c_wvs2 += cos_sim * context_wvs[b, :, c]
                new_context_wvs = T.set_subtensor(new_context_wvs[b, :, 2], c_wvs / total_cos_sim)
                new_context_wvs = T.set_subtensor(new_context_wvs[b, :, 3], c_wvs2 / total_cos_sim2)
            new_neg_wvs = neg_wvs[:, :, :4 * 10]
        elif mode == 3:
            new_context_wvs = T.zeros_like(context_wvs[:, :, :2], dtype=theano.config.floatX)
            for b in range(self.batch_size):
                c_wvs = numpy.zeros((self.img_w,), dtype=theano.config.floatX)
                total_cos_sim = 0
                c_wvs2 = numpy.zeros((self.img_w,), dtype=theano.config.floatX)
                total_cos_sim2 = 0
                for c in range(2 + self.ctx_size*4):
                    if c < 1 + self.ctx_size*2:
                        cos_sim = T.exp(T.dot(self.input[b], context_wvs[b, :, c]))
                        total_cos_sim += cos_sim
                        c_wvs += cos_sim * context_wvs[b, :, c]
                    else:
                        cos_sim = T.exp(T.dot(self.input[b], context_wvs[b, :, c]))
                        total_cos_sim2 += cos_sim
                        c_wvs2 += cos_sim * context_wvs[b, :, c]
                new_context_wvs = T.set_subtensor(new_context_wvs[b, :, 0], c_wvs / total_cos_sim)
                new_context_wvs = T.set_subtensor(new_context_wvs[b, :, 1], c_wvs2 / total_cos_sim2)
            new_neg_wvs = neg_wvs[:, :, :2 * 10]
        elif mode == 4:
            new_context_wvs = T.zeros_like(context_wvs[:, :, :1], dtype=theano.config.floatX)
            for b in range(self.batch_size):
                c_wvs = numpy.zeros((self.img_w,), dtype=theano.config.floatX)
                total_cos_sim = 0
                for c in range(2 + self.ctx_size*4):
                    cos_sim = T.exp(T.dot(self.input[b], context_wvs[b, :, c]))
                    total_cos_sim += cos_sim
                    c_wvs += cos_sim * context_wvs[b, :, c]
                new_context_wvs = T.set_subtensor(new_context_wvs[b, :, 0], c_wvs / total_cos_sim)
            new_neg_wvs = neg_wvs[:, :, :10]

        # Minimize cross-entropy loss function
        results, _ = theano.scan(lambda input, context_wv:
                                 T.log(sigmoid(T.dot(input, context_wv))),
                                 outputs_info=None,
                                 sequences=[self.input, new_context_wvs])
        neg_results, _ = theano.scan(lambda input, neg_wv:
                                     T.log(sigmoid(-T.dot(input, neg_wv))),
                                     outputs_info=None, sequences=[self.input, new_neg_wvs])

        return -(T.sum(results, dtype=theano.config.floatX) +
                 T.sum(neg_results, dtype=theano.config.floatX))


    def cost_depsp(self, context_idx, context_msk, neg_idx, mode):
        # context_idx: (B, S)
        # context_wvs: (B, H, S)
        context_wvs = self.words[context_idx.flatten()].reshape(
            (self.batch_size, self.max_l, self.img_w)).transpose((0, 2, 1))
        neg_wvs = self.words[neg_idx.flatten()].reshape(
            (self.batch_size, self.max_l*10, self.img_w)).transpose((0, 2, 1))
        new_context_wvs = context_wvs
        new_context_msk = context_msk
        new_neg_wvs = neg_wvs
        new_neg_msk = context_msk.repeat(10, axis=1)

        if mode == 1:
            new_context_wvs = T.zeros_like(context_wvs[:, :, :1],
                                           dtype=theano.config.floatX)
            for b in range(self.batch_size):
                cos_sim = T.exp(T.dot(self.input[b], context_wvs[b, :, :]))
                cos_sim =  cos_sim * context_msk[b, :]
                cos_sim = cos_sim / cos_sim.sum()
                new_context_wvs = T.set_subtensor(
                    new_context_wvs[b, :, 0],
                    T.dot(context_wvs[b, :, :], cos_sim))
            new_neg_wvs = neg_wvs[:, :, :10]
            new_context_msk = new_context_msk[:, :1]
            new_neg_msk = new_neg_msk[:, :10]

        # Minimize cross-entropy loss function
        results, _ = theano.scan(
            lambda input, context_wv, mask:
            T.log(sigmoid(T.dot(input, context_wv)))*mask,
            outputs_info=None,
            sequences=[self.input, new_context_wvs, new_context_msk])
        neg_results, _ = theano.scan(
            lambda input, neg_wv, mask:
            T.log(sigmoid(-T.dot(input, neg_wv)))*mask,
            outputs_info=None,
            sequences=[self.input, new_neg_wvs, new_neg_msk])

        res = -(T.sum(results, dtype=theano.config.floatX) +
                 T.sum(neg_results, dtype=theano.config.floatX))
        return  res / T.sum(context_msk)
