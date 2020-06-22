import sys
sys.path.append('../')
import os
from tqdm import tqdm
from rouge_score import rouge_scorer
from IR import lev_translate, token2index, index2token
import pickle
from util import integrated_score, read_testset


METRIC_LIST = ['BLEU3', 'BLEU4', 'CIDEr', 'ROUGE_L']


if __name__ == '__main__':
    f = open('../simplified_dataset/train_CODENN_data.pkl', 'rb')
    code_train, comment_train, code_voc, comment_voc = pickle.load(f)
    f.close()
    test_inputs, test_outputs = read_testset('../simplified_dataset/simplified_test.json')
    f_parameter = open("parameters", "a")
 
    total_score = dict()
    for metric in METRIC_LIST:
        total_score[metric] = 0

    for index, test in enumerate(tqdm(test_inputs[10000:])):
        seq = token2index(test, code_voc)
        comment_index = lev_translate(seq, code_train)
        predict = comment_train[comment_index]
        predict = ' '.join([i for i in index2token(predict, comment_voc) if i != "<PAD>"])
        for metric in METRIC_LIST:
            score = integrated_score(metric, test_outputs[index], predict)
            total_score[metric] += score
        if (index%1000) == 0:
            print(index, total_score)
    
    f_parameter.write("10000-\n")
    for metric in METRIC_LIST:
        total_score[metric] = total_score[metric] / len(test_inputs)    
        f_parameter.write(metric+"="+str(round(total_score[metric], 4))+"\n")
        f_parameter.flush()
    print(metric+"="+str(round(total_score[metric], 4)))

    f_parameter.close()
