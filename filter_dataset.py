import json
import os
import javalang
import nltk
from tqdm import tqdm

# Usage:
# remove the noisy data in deepcom's dataset, and create a smaller dataset


def is_invalid_method(code: str, comment: str):
    tokens_parse = javalang.tokenizer.tokenize(code)
    token_len = len(list(tokens_parse))

    if token_len > 350 or len(code.split('\n')) > 40:
        return True
    if len(comment.split('.')) != 1 or len(nltk.word_tokenize(comment)) > 30:
        return True
    return False


path_list = [
    './DeepCom_data/train.json',
    './DeepCom_data/valid.json',
    './DeepCom_data/test.json'
]
SAVE_PATH = './simplified_dataset'

deepCom_data = []
for path in path_list:
    deepcom_file = open(path)
    deepCom_data.extend(deepcom_file.readlines())
    deepcom_file.close()

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

new_training_set = open(SAVE_PATH+'/simplified_train.json', "w")
new_testing_set = open(SAVE_PATH+'/simplified_test.json', "w")

# ================= filter DeepCom's dataset ====================
print('DeepCom total data units: '+str(len(deepCom_data)))
new_data = []
for pair in tqdm(deepCom_data):
    pair = json.loads(pair)
    if is_invalid_method(pair['code'], pair['nl']):
        continue
    new_data.append(json.dumps(pair))
# ===============================================================


# =========== divide dataset into training set and testing set ==============
print('Quantity of data units: '+str(len(new_data)))
TRAIN_INDEX = int(len(new_data)*0.9)
TEST_INDEX = int(len(new_data)-1)
train_data = new_data[:TRAIN_INDEX]
test_data = new_data[TRAIN_INDEX+1:TEST_INDEX]
# ===========================================================================


# ================== write the new dataset ======================
for row in tqdm(train_data):
    new_training_set.write(row+'\n')
new_training_set.close()
print('finish writing the simplified train data')

for row in tqdm(test_data):
    new_testing_set.write(row+'\n')
new_testing_set.close()
print('finish writing the simplified test data')
# ===============================================================


# ========= write the new testing set according to LOC =================
new_testing_set_loc = [None, None, None, None, None]
new_testing_set_loc[0] = open(SAVE_PATH+'/simplified_test_0_10.json', "w")
new_testing_set_loc[1] = open(SAVE_PATH+'/simplified_test_10_20.json', "w")
new_testing_set_loc[2] = open(SAVE_PATH+'/simplified_test_20_30.json', "w")
new_testing_set_loc[3] = open(SAVE_PATH+'/simplified_test_30_40.json', "w")
new_testing_set_loc[4] = new_testing_set_loc[3]
cnt = [0, 0, 0, 0, 0]

for row in tqdm(test_data):
    pair = json.loads(row)
    loc = len(pair['code'].split('\n'))
    new_testing_set_loc[loc//10].write(row+'\n')
    cnt[loc//10] += 1

for file in new_testing_set_loc:
    file.close()
print('[0-10, 10-20, 20-30, 30-40, 40]:', cnt)
print('finish clustering simplified test data by LOC')
# ======================================================================
