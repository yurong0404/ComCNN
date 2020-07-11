import os

MODE = "DeepCom"  #CODE-NN(ARCH=CODE-NN) or DeepCom or ComCNN or Hybrid-DeepCom
ARCH = "DeepCom"  # lstm_lstm or cnnlstm_lstm or cnnbilstm_lstm or CODE-NN or Hybrid-DeepCom or DeepCom
BATCH_SIZE = 10
EMBEDDING_DIM = 256
UNITS = 256
FILTERS=256  # hyper-parameter of CNN

os.environ["CUDA_VISIBLE_DEVICES"] = '0'