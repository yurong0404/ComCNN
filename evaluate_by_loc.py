import os
from util import *
from model import *
from config import *
from tqdm import tqdm

METRIC_LIST = ['BLEU3', 'BLEU4', 'CIDEr', 'ROUGE_L']
PREDICT_METHOD_LIST = ['greedy', 'beam_3', 'beam_5']
DATASET_PATH = [
    './simplified_dataset/simplified_test_0_10.json',
    './simplified_dataset/simplified_test_10_20.json',
    './simplified_dataset/simplified_test_20_30.json',
    './simplified_dataset/simplified_test_30_40.json'
]


if __name__ == '__main__':
    train_data = read_train_pkl()
    encoder, decoder = create_model(
        train_data['code_voc_size'],
        train_data['com_voc_size'],
        train_data['max_length_code']
    )
    encoder, decoder = restore_model(encoder, decoder)

    print('arch:', ARCH)
    print("Reading model...")

    checkpoint_dir = get_checkpoint_dir()
    log_file = open(checkpoint_dir+"/performance_by_loc", "a")
    for index, dataset in enumerate(DATASET_PATH):
        test_data = read_testset(path=dataset)
        log_file.write('LOC:'+str(index*10)+'_'+str(index*10+10)+'\n')
        print('\nLOC:'+str(index*10)+'_'+str(index*10+10))
        for method in PREDICT_METHOD_LIST:
            log_file.write('    '+method+'\n')
            log_file.flush()
            print(method+'\n')
            if method == 'beam_3' or method == 'beam_5':
                beam_k = int(method.split('_')[1])
            else:
                beam_k = 1

            total_score = dict()
            for metric in METRIC_LIST:
                total_score[metric] = 0

            for data in tqdm(test_data):
                predict = integrated_prediction(
                    data['code'],
                    encoder,
                    decoder,
                    train_data,
                    beam_k,
                    method,
                )
                for metric in METRIC_LIST:
                    score = integrated_score(metric, data['comment'], predict)
                    total_score[metric] += score

            for metric in METRIC_LIST:
                total_score[metric] = total_score[metric] / len(test_data)
                log_file.write('        '+metric+"="+str(round(total_score[metric], 4))+"\n")
                log_file.flush()
                print(metric+"="+str(round(total_score[metric], 4)))

    log_file.close()
