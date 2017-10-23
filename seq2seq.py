import numpy as np
import theano
import theano.tensor as tensor
from theano.tensor.nnet.nnet import sigmoid

import utils
import negsampling


class LSTM(object):
    def __init__(self, hidden_size):

        self.hidden_size = hidden_size

        # lstm W matrixes, Wf, Wi, Wo, Wc respectively, all config.floatX type
        self.W = theano.shared(name="W", value=utils.init_norm(self.hidden_size, 4 * self.hidden_size),
                               borrow=True)
        # lstm U matrixes, Uf, Ui, Uo, Uc respectively, all config.floatX type
        self.U = theano.shared(name="U", value=utils.init_norm(self.hidden_size, 4 * self.hidden_size),
                               borrow=True)
        # lstm b vectors, bf, bi, bo, bc respectively, all config.floatX type
        self.b = theano.shared(name="b", value=np.zeros(4 * self.hidden_size, dtype=theano.config.floatX), borrow=True)

        self.params = [self.W, self.U, self.b]

    def forward(self, inputs, mask, h0=None, C0=None):
        """
        param inputs: #(max_sent_size, batch_size, hidden_size).
        inputs: state_below
        """
        if inputs.ndim == 3:
            batch_size = inputs.shape[1]
        else:
            batch_size = 1

        if h0 == None:
            h0 = tensor.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)
        if C0 == None:
            C0 = tensor.alloc(np.asarray(0., dtype=theano.config.floatX), batch_size, self.hidden_size)

        def _step(m, X, h_, C_, W, U, b):
            XW = tensor.dot(X, W)  # (batch_size, 4*hidden_size)
            h_U = tensor.dot(h_, U)  # (batch_size, 4*hidden_size)
            # before activation,       (batch_size, 4*hidden_size)
            bfr_actv = XW + h_U + b

            f = tensor.nnet.sigmoid(bfr_actv[:, 0:self.hidden_size])  # forget gate (batch_size, hidden_size)
            i = tensor.nnet.sigmoid(
                bfr_actv[:, 1 * self.hidden_size:2 * self.hidden_size])  # input gate (batch_size, hidden_size)
            o = tensor.nnet.sigmoid(
                bfr_actv[:, 2 * self.hidden_size:3 * self.hidden_size])  # output  gate (batch_size, hidden_size)
            Cp = tensor.tanh(
                bfr_actv[:, 3 * self.hidden_size:4 * self.hidden_size])  # candi states (batch_size, hidden_size)

            C = i * Cp + f * C_
            C = m[:, None] * C + (1.0 - m)[:, None] * C_

            h = o * tensor.tanh(C)
            h = m[:, None] * h + (1.0 - m)[:, None] * h_
            h, C = tensor.cast(h, theano.config.floatX), tensor.cast(C, theano.config.floatX)
            return h, C

        outputs, updates = theano.scan(
            fn=_step,
            sequences=[mask, inputs],
            outputs_info=[h0, C0],
            non_sequences=[self.W, self.U, self.b]
        )

        hs, Cs = outputs
        return hs, Cs


class Seq2Seq(object):
    def __init__(self, words, input, voca_size, hidden_size, max_l, batch_size,
                 neg_num, lstm_layers_num, inputdir, neg_table, learning_rate=0.2, ):
        self.words = words
        self.input = input
        self.voca_size = voca_size
        self.hidden_size = hidden_size
        self.max_l = max_l
        self.batch_size = batch_size
        self.neg_num = neg_num
        self.lstm_layers_num = lstm_layers_num
        self.learning_rate = learning_rate
        self._train = None
        self._utter = None
        self.params = []
        self.encoder_lstm_layers = []
        self.decoder_lstm_layers = []
        self.hos = []
        self.Cos = []
        self.table = neg_table

    def cost(self, decoderInputs, decoderMask, decoderTarget, neg_idx):
        # self.linear = theano.shared(
        #     name="Linear",
        #     value=utils_seq2seq.init_norm(self.hidden_size, self.voca_size),
        #     borrow=True
        # )
        # self.params += [self.linear]  # concatenate

        # (max_sent_size, batch_size, hidden_size)
        # state_below = self.words[encoderInputs.flatten()].reshape((encoderInputs.shape[0], encoderInputs.shape[1], self.hidden_size))
        # for _ in range(self.lstm_layers_num):
        #     enclstm = LSTM(self.hidden_size)
        #     self.encoder_lstm_layers += enclstm,    #append
        #     self.params += enclstm.params    #concatenate
        #     hs, Cs = enclstm.forward(state_below, encoderMask)
        #     self.hos += hs[-1],
        #     self.Cos += Cs[-1],
        #     state_below = hs
        # TODO: check how to handle Cos
        self.hos.append(self.input)
        self.Cos.append(tensor.zeros_like(self.input, dtype=theano.config.floatX))

        state_below = self.words[decoderInputs.flatten()].reshape(
            (decoderInputs.shape[0], decoderInputs.shape[1], self.hidden_size))
        for i in range(self.lstm_layers_num):
            declstm = LSTM(self.hidden_size)
            self.decoder_lstm_layers += declstm,  # append
            self.params += declstm.params  # concatenate
            ho, Co = self.hos[i], self.Cos[i]
            state_below, Cs = declstm.forward(state_below, decoderMask, ho, Co)
        decoder_lstm_outputs = state_below

        di, dm, dt = tensor.imatrices(3)  # place holders
        #####################################################
        #####################################################

        target_wvs = self.words[decoderTarget.flatten()].reshape(
            (self.max_l, self.batch_size, self.hidden_size))
        neg_wvs = self.words[neg_idx.flatten()].reshape(
            (self.max_l, self.batch_size, self.neg_num, self.hidden_size))

        # m1, m2 must have same shape (self.batch_size, self.hidden_size)
        def _rowwise_dot(m1, m2):
            return tensor.sum(m1 * m2, axis=1).reshape((self.batch_size, 1))

        target_cost, _ = theano.scan(
            fn=lambda sent_emb, word_emb, mask:
            tensor.log(sigmoid(_rowwise_dot(sent_emb, word_emb))) * mask,
            outputs_info=None,
            sequences=[decoder_lstm_outputs, target_wvs, decoderMask]
        )

        def _sample_aggregation(sent_emb, word_emb, mask):
            res = tensor.zeros((self.batch_size, 1), dtype=theano.config.floatX)
            for k in range(self.neg_num):
                res += tensor.log(sigmoid(-1 * _rowwise_dot(sent_emb, word_emb[:, k, :]))),
            return res * mask

        neg_cost, _ = theano.scan(
            fn=_sample_aggregation,
            outputs_info=None,
            sequences=[decoder_lstm_outputs, neg_wvs, decoderMask]
        )

        return -(target_cost.sum() + neg_cost.sum()) / decoderMask.sum()

        '''
        linear_outputs = tensor.dot(decoder_lstm_outputs, self.linear)
        softmax_outputs, updates = theano.scan(
            fn=lambda x: tensor.nnet.softmax(x),
            sequences=[linear_outputs],
        )

        def _NLL(pred, y, m):
            return -m * tensor.log(pred[tensor.arange(decoderInputs.shape[1]), y])

        costs, updates = theano.scan(fn=_NLL, sequences=[softmax_outputs, decoderTarget, decoderMask])
        return costs.sum() / decoderMask.sum()
        '''
