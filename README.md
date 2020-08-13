# ComCNN

這是一個將程式碼透過深度學習，自動翻譯成註解的研究，目標程式碼以function為單位，而註解會說明該段function的程式邏輯

This is a study that automatically translates the code into comments by using deep learning. The target code are functions, and the comments will explain the function behavior.

TensorFlow version: 2.1.0<br>
TensorFlow.keras version: 2.2.4-tf<br>
***

## filter_dataset.py

The dataset in this study is derived from DeepCom. This script will remove the noisy data in DeepCom's dataset and save it as a smaller dataset (simplified_train.json and simplified_test.json).
***
## readdata.py
It reads the dataset, and preprocesses the training data. The data-preprocess is time-consuming. To save the preprocessing time, it writes the training data into pickle file. Once we need to preprocess the training data again, read the pickle file! No more preprocess is needed.
***
## train.py
This script trains our proposed model, and config.py can set the architecture

***
## config.py

This file set the global constant variables for whole project.
***
## predict.py
這檔案讀取已訓練好的模型，並且使用該模型將程式碼翻譯成註解

This script restores the pre-trained model and translate given codes to the comments.
***
## evaluate.py
這檔案讀取已訓練好的模型，並且使用bleu3、bleu4, CIDEr, ROUGE-L評測該模型

This script restores the trained model and evaluates the model with bleu3, bleu4, CIDEr, ROUGE-L.
***

## evaluate_by_loc.py
這個檔案功能與evaluate.py類似，只是分別針對特定行碼數的testing data進行評測

This script is similar to evaluate.py. It evaluates different testing sets according to LOC levels.
***
## util.py
這裡面有很多小function

There are a lot of functions in the file.
***
