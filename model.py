import tensorflow as tf
import numpy as np
from param import *

def lstm(units):
    return tf.keras.layers.LSTM(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = lstm(self.enc_units)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state = hidden)
        return output, state_h, state_c
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = lstm(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden[1], 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        
        return x, state_h, state_c
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units)), tf.zeros((self.batch_sz, self.dec_units))

class BidirectionalEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(BidirectionalEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = lstm(self.enc_units)
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm) 
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.bilstm(x, initial_state = hidden)
        return output, forward_h, forward_c, backward_h, backward_c
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)), \
                tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))

class BidirectionalDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(BidirectionalDecoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = lstm(self.dec_units)
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.W3 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        forward_hidden_c = tf.expand_dims(hidden[1], 1)
        backward_hidden_c = tf.expand_dims(hidden[3], 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(forward_hidden_c) + self.W3(backward_hidden_c)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, forward_h, forward_c, backward_h, backward_c = self.bilstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        
        return x, forward_h, forward_c, backward_h, backward_c
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units)), tf.zeros((self.batch_sz, self.dec_units))

class cnnEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, filters, batch_sz, max_length_inp):
        super(cnnEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.kernel_size = 3
        self.strides = 1
        self.enc_units = filters
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.cnn = tf.keras.layers.Conv1D(filters=self.enc_units,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        activation='tanh',
                                        input_shape=(max_length_inp, embedding_dim))
        self.pool = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)
        # output shape = (?, filters)  ? = (max_length_inp-(kernel_sz-1))//strides//pool_strides
        
    def call(self, x):
        x = self.embedding(x)
        x = self.cnn(x)
        x = self.pool(x)
        return x
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))


class cnnBidirectionalEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, filters, batch_sz, max_length_inp):
        super(cnnEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.kernel_size = 3
        self.strides = 1
        self.enc_units = filters
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.cnn = tf.keras.layers.Conv1D(filters=self.enc_units,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        activation='tanh',
                                        input_shape=(max_length_inp, embedding_dim))
        self.pool = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)
        # output shape = (?, filters)  ? = (max_length_inp-(kernel_sz-1))//strides//pool_strides

        self.lstm = lstm(self.enc_units)
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm) 
        
    def call(self, x, hidden):
        x = self.embedding(x)
        x = self.cnn(x)
        output, forward_h, forward_c, backward_h, backward_c = self.bilstm(x, initial_state = (forward_h, forward_c, backward_h, backward_c))
        return output, forward_h, forward_c, backward_h, backward_c
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)