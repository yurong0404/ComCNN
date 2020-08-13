import os
from util import *
from model import *
from config import *
from tqdm import tqdm
from rouge_score import rouge_scorer
import tempfile


METRIC_LIST = ['BLEU3', 'BLEU4', 'CIDEr', 'ROUGE_L']
PREDICT_METHOD_LIST = ['greedy', 'beam_3', 'beam_5']


if __name__ == '__main__':
    train_data = read_train_pkl()
    encoder, decoder = create_model(
        train_data['code_voc_size'],
        train_data['com_voc_size'],
        train_data['max_length_code']
    )
    encoder, decoder = restore_model(encoder, decoder)
    test_data = read_testset()

    print('arch:', ARCH)
    print("Reading model...")

    checkpoint_dir = get_checkpoint_dir()
    log_file = open(checkpoint_dir+"/parameters", "a")

    for method in PREDICT_METHOD_LIST:
        log_file.write(method+'\n')
        log_file.flush()
        print('\n'+method+'\n')

        if method == 'beam_3' or method == 'beam_5':
            beam_k = int(method.split('_')[1])
        elif method == 'greedy':
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
            log_file.write('    '+metric+"="+str(round(total_score[metric], 4))+"\n")
            log_file.flush()
            print(metric+"="+str(round(total_score[metric], 4)))

    log_file.close()
