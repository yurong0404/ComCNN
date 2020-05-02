import os

MODE = "CODE-NN"  #CODE-NN(ARCH=CODE-NN) or DeepCom(ARCH=lstm_lstm) or ComCNN
ARCH = "CODE-NN"  # lstm_lstm  or cnn_lstm or cnn_bilstm or cnnlstm_lstm or cnnbilstm_lstm or CODE-NN
BATCH_SIZE = 32
EMBEDDING_DIM = 256
UNITS = 256
FILTERS=256  # hyper-parameter of CNN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'