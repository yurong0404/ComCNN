import tensorflow as tf
import numpy as np
from config import *

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
        output = output[:,:,:self.enc_units] * output[:,:,self.enc_units:]
        return output, forward_h, forward_c
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)), \
                tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))


class cnnbilstmEncoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, filters, batch_sz, max_length_inp):
        super(cnnbilstmEncoder, self).__init__()
        self.batch_sz = batch_sz
        self.kernel_size = 3
        self.strides = 1
        self.enc_units = filters
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.cnn = tf.keras.layers.Conv1D(filters=self.enc_units,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        input_shape=(max_length_inp, embedding_dim))
        self.pool = tf.keras.layers.MaxPool1D(pool_size = 2, strides = 2)
        # output shape = (?, filters)  ? = (max_length_inp-(kernel_sz-1))//strides//pool_strides
        self.lstm = lstm(self.enc_units)
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        #_, state_h, state_c = self.lstm(x, initial_state = hidden)
        _, forward_h, forward_c, backward_h, backward_c = self.bilstm(x, initial_state = hidden)
        state_h = forward_h * backward_h
        state_c = forward_c * backward_c
        x = self.cnn(x)
        x = self.pool(x)
        
        return x, state_h, state_c
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units)), \
                tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))



class codennDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, code_vocab_size):
        super(codennDecoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.code_embedding = tf.keras.layers.Embedding(code_vocab_size, embedding_dim)
        self.lstm = lstm(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, code):
        hidden_with_time_axis = tf.expand_dims(hidden[0], 1)
        code = self.code_embedding(code)
        score = tf.math.exp(hidden_with_time_axis * code)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * code
        context_vector = tf.reduce_sum(context_vector, axis=1)
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        output = tf.nn.tanh(self.W1(state_h) + self.W2(context_vector))
        x = self.fc(output)
        
        return x, state_h, state_c
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units)), tf.zeros((self.batch_sz, self.dec_units))



def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)