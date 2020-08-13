import tensorflow as tf
import numpy as np
from config import *


def lstm(units):
    return tf.keras.layers.LSTM(
        units,
        return_sequences=True,
        return_state=True,
        recurrent_activation='sigmoid',
        recurrent_initializer='glorot_uniform'
    )


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss)


class lstmEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(lstmEncoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = lstm(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return output, state_h, state_c

    def initialize_hidden_state(self, batch_sz):
        return [tf.zeros((batch_sz, self.enc_units)), tf.zeros((batch_sz, self.enc_units))]


class cnnlstmEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, filters, max_length_code):
        super(cnnlstmEncoder, self).__init__()
        self.kernel_size = 3
        self.strides = 1
        self.enc_units = filters
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.cnn = tf.keras.layers.Conv1D(
            filters=self.enc_units,
            kernel_size=self.kernel_size,
            strides=self.strides,
            input_shape=(max_length_code, embedding_dim)
        )
        self.pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        # output shape = (?, filters)  ? = (max_length_inp-(kernel_sz-1))//strides//pool_strides
        self.lstm = lstm(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        _, state_h, state_c = self.lstm(x, initial_state=hidden)
        x = self.cnn(x)
        x = self.pool(x)

        return x, state_h, state_c

    def initialize_hidden_state(self, batch_sz):
        return [tf.zeros((batch_sz, self.enc_units)), tf.zeros((batch_sz, self.enc_units))]


class cnnbilstmEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, filters, max_length_code):
        super(cnnbilstmEncoder, self).__init__()
        self.kernel_size = 3
        self.strides = 1
        self.enc_units = filters
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.cnn = tf.keras.layers.Conv1D(
            filters=self.enc_units,
            kernel_size=self.kernel_size,
            strides=self.strides,
            input_shape=(max_length_code, embedding_dim)
        )
        self.pool = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        # self.pool = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)
        # output shape = (?, filters)  ? = (max_length_code - (kernel_sz-1))//strides//pool_strides
        self.lstm = lstm(self.enc_units)
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm)

    def call(self, x, hidden):
        x = self.embedding(x)
        _, forward_h, forward_c, backward_h, backward_c = self.bilstm(x, initial_state=hidden)
        state_h = forward_h * backward_h
        state_c = forward_c * backward_c
        x = self.cnn(x)
        x = self.pool(x)

        return x, state_h, state_c

    def initialize_hidden_state(self, batch_sz):
        return [tf.zeros((batch_sz, self.enc_units)), tf.zeros((batch_sz, self.enc_units)),
                tf.zeros((batch_sz, self.enc_units)), tf.zeros((batch_sz, self.enc_units))]


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = lstm(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden[0], 1)
        # shape of score (batch, max_length, 1)
        score = self.V(tf.math.exp(hidden_with_time_axis * enc_output))
        attention_weights = tf.nn.softmax(score, axis=1)
        # shape of context_vector (batch, max_length, dec_units)
        context_vector = attention_weights * enc_output
        # shape of context_vector after sum (batch, dec_units)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        output = tf.nn.tanh(self.W1(state_h) + self.W2(context_vector))
        x = self.fc(output)
        return x, state_h, state_c

    def initialize_hidden_state(self, batch_sz):
        return tf.zeros((batch_sz, self.dec_units)), tf.zeros((batch_sz, self.dec_units))
