import json
import javalang
import nltk
import os
import random
from tqdm import tqdm

# Usage:
# Read DeepCom's dataset, then filter the data through specific rules, and finally save the data as a smaller dataset

def is_invalid_method(code: str, nl: str):   
    tokens_parse = javalang.tokenizer.tokenize(code)
    token_len = len(list(tokens_parse))
    
    if token_len > 350 or len(code.split('\n')) > 40:
        return True
    if len(nl.split('.')) != 1 or len(nltk.word_tokenize(nl)) > 30:
        return True
    else :
        return False

path_list = ['./DeepCom_data/train.json', './DeepCom_data/valid.json', './DeepCom_data/test.json']
save_path = './simplified_dataset'

inputs = []
for path in path_list:
    input_file = open(path)
    inputs.extend(input_file.readlines())
    input_file.close()
outputs = []

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
output_train_file = open(save_path+'/simplified_train.json', "w")
output_test_file = open(save_path+'/simplified_test.json', "w")

# ================= filter DeepCom's dataset ====================
print('Original total: '+str(len(inputs)))
for pair in tqdm(inputs):
    pair = json.loads(pair)
    if is_invalid_method(pair['code'], pair['nl']):
        continue
    outputs.append(json.dumps(pair))
# ===============================================================

#random.shuffle(outputs)

# ============== divide dataset into training set and testing set =================
print('Final total: '+str(len(outputs)))
print('Data shuffle complete')
train_index = int(len(outputs)*0.9)
test_index = int(len(outputs)-1)
train_output = outputs[:train_index]
test_output = outputs[train_index+1:test_index]
# =================================================================================


# ================== write the new dataset ======================
for row in tqdm(train_output):
    output_train_file.write(row+'\n')
output_train_file.close()
print('simplified train data finish writing')
for row in tqdm(test_output):
    output_test_file.write(row+'\n')
output_test_file.close()
print('simplified test data finish writing')
# ===============================================================


# ================== write the new training set according to LOC ======================
output_train_file_line = [0, 0, 0, 0, 0]
output_train_file_line[0] = open(save_path+'/simplified_train_0_10.json', "w")
output_train_file_line[1] = open(save_path+'/simplified_train_10_20.json', "w")
output_train_file_line[2] = open(save_path+'/simplified_train_20_30.json', "w")
output_train_file_line[3] = open(save_path+'/simplified_train_30_40.json', "w")
output_train_file_line[4] = output_train_file_line[3]

for row in tqdm(train_output):
    pair = json.loads(row)
    loc = len(pair['code'].split('\n'))
    output_train_file_line[loc//10].write(row+'\n')

for file in output_train_file_line:
    file.close()
print('simplified training data finish clustering by LOC')
# =====================================================================================


# ================== write the new testing set according to LOC ======================
output_test_file_line = [0, 0, 0, 0, 0]
output_test_file_line[0] = open(save_path+'/simplified_test_0_10.json', "w")
output_test_file_line[1] = open(save_path+'/simplified_test_10_20.json', "w")
output_test_file_line[2] = open(save_path+'/simplified_test_20_30.json', "w")
output_test_file_line[3] = open(save_path+'/simplified_test_30_40.json', "w")
output_test_file_line[4] = output_test_file_line[3]
cnt = [0, 0, 0, 0, 0]

for row in tqdm(test_output):
    pair = json.loads(row)
    loc = len(pair['code'].split('\n'))
    output_test_file_line[loc//10].write(row+'\n')
    cnt[loc//10] += 1

for file in output_test_file_line:
    file.close()
print(cnt)
print('simplified test data finish clustering by LOC')
# =====================================================================================