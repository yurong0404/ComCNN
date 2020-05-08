import os

MODE = "ComCNN"  #CODE-NN(ARCH=CODE-NN) or DeepCom(ARCH=lstm_lstm) or ComCNN
ARCH = "cnn_lstm"  # lstm_lstm or cnn_lstm or bilstm_lstm or CODE-NN
BATCH_SIZE = 40
EMBEDDING_DIM = 256
UNITS = 256
FILTERS=256  # hyper-parameter of CNN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'