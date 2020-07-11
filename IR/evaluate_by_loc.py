import sys
sys.path.append('../')
import os
from tqdm import tqdm
from rouge_score import rouge_scorer
from IR import lev_translate, token2index, index2token
import pickle
from util import integrated_score, read_testset


METRIC_LIST = ['BLEU3', 'BLEU4', 'CIDEr', 'ROUGE_L']

DATASET_PATH = [
    '../simplified_dataset/simplified_test_0_10.json',
    '../simplified_dataset/simplified_test_10_20.json',
    '../simplified_dataset/simplified_test_20_30.json',
    '../simplified_dataset/simplified_test_30_40.json'
]


if __name__ == '__main__':
    f = open('../simplified_dataset/train_CODENN_data.pkl', 'rb')
    code_train, comment_train, code_voc, comment_voc = pickle.load(f)
    f.close()

    f_parameter = open("performance_by_loc_30_40", "a")
    #for index, dataset in enumerate(DATASET_PATH):
    test_inputs, test_outputs = read_testset('../simplified_dataset/simplified_test_30_40.json')

    total_score = dict()

    for metric in METRIC_LIST:
        total_score[metric] = 0

    for index, test in enumerate(tqdm(test_inputs[320:])):
        seq = token2index(test, code_voc)
        comment_index = lev_translate(seq, code_train)
        predict = comment_train[comment_index]
        predict = ' '.join([i for i in index2token(predict, comment_voc) if i != "<PAD>"])
        for metric in METRIC_LIST:
            score = integrated_score(metric, test_outputs[index], predict)
            total_score[metric] += score
        #if (index%1000) == 0:
        #    print(index, total_score)
    
    f_parameter.write("320-\n")
    for metric in METRIC_LIST:
        total_score[metric] = total_score[metric] / len(test_inputs)    
        f_parameter.write(metric+"="+str(round(total_score[metric], 6))+"\n")
        print(metric+"="+str(round(total_score[metric], 6)))
        f_parameter.flush()

    f_parameter.close()
