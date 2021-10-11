# ComCNN

此專案是我在清華資工所的碩士論文<br>
This is my master thesis project in National Tsing Hua University.<br><br>
Download PDF: ComCNN_thesis.pdf (Available after 2023/08/16)

***
這是一個將程式碼透過深度學習，自動翻譯成註解的研究，目標程式碼以function為單位，而註解會說明該段function的程式邏輯

This is a study that automatically translates the code into comments by using deep learning. The target code are functions, and the comments will explain the function behavior.

TensorFlow version: 2.1.0<br>
TensorFlow.keras version: 2.2.4-tf<br>
***

## filter_dataset.py
The dataset in this study is derived from DeepCom. This script will remove the noisy data in DeepCom's dataset and save it as a smaller dataset (simplified_train.json and simplified_test.json).

## readdata.py
It reads the dataset, and preprocesses the training data. The data-preprocess is time-consuming. To save the preprocessing time, it writes the training data into pickle file. Once we need to preprocess the training data again, read the pickle file! No more preprocess is needed.

## train.py
This script trains our proposed model, and config.py can set the architecture

## config.py
This file set the global constant variables for whole project.

## predict.py
This script restores the pre-trained model and translate given codes to the comments.

## evaluate.py
This script restores the trained model and evaluates the model with bleu3, bleu4, CIDEr, ROUGE-L.

## evaluate_by_loc.py
This script is similar to evaluate.py. It evaluates different testing sets according to LOC levels.

## util.py
There are a lot of functions in the file.
***

## Create new dataset from DeepCom's dataset 
(DeepCom's dataset need to be under the "./DeepCom_data/")
```bash
$ python3 filter_dataset.py
```
After the script finishs running, new dataset will be created under the "./simplified_dataset/"

***

## Preprocess the training data
```bash
$ python3 readdata.py
```
Training data will be saved as a pickle file at "./simplified_dataset/xxx.pkl"
***

## Train the model
```bash
$ python3 train.py
```
After training, the trained model will be saved under "./training_checkpoints/"
***
## Use the trained model to predict comments
```bash
$ python3 predict.py
```
***
## Evaluate the model
```bash
$ python3 evaluate.py
$ python3 evaluate_by_loc.py
```
After evaluating, the results will be saved at "./training_checkpoints/model_name/parameters" and "./training_checkpoints/model_name/performance_by_loc"
***
## Change configuration
please edit the config.py file 
