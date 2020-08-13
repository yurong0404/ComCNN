import os

ARCH = "cnnbilstm_lstm"  # lstm_lstm or cnnlstm_lstm or cnnbilstm_lstm
BATCH_SIZE = 40
EMBEDDING_DIM = 256
UNITS = 256
FILTERS = 256  # hyper-parameter of CNN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
