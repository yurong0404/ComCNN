import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from util import *
from model import *
from param import *
from tqdm import tqdm
from rouge_score import rouge_scorer
import tempfile


METRIC_LIST = ['BLEU3', 'BLEU4', 'CIDEr', 'ROUGE_L']
PREDICT_METHOD_LIST = ['greedy', 'beam_3', 'beam_5']


if __name__ == '__main__':
    code_train, comment_train, code_voc, comment_voc = read_pkl()
    vocab_inp_size = len(code_voc)
    vocab_tar_size = len(comment_voc)
    max_length_inp = max(len(t) for t in code_train)
    max_length_targ = max(len(t) for t in comment_train)
    
    encoder, decoder= create_encoder_decoder(vocab_inp_size, vocab_tar_size, max_length_inp)

    encoder, decoder = read_model(encoder, decoder)
    test_inputs, test_outputs = read_testset('./simplified_dataset/simplified_test.json')

    print('mode:', MODE, ', arch:', ARCH)
    print("Reading model...")

    
    checkpoint_dir = getCheckpointDir()
    f_parameter = open(checkpoint_dir+"/parameters", "a")
 
    for method in PREDICT_METHOD_LIST:
        f_parameter.write(method+'\n')
        f_parameter.flush()
        print('\n'+method+'\n')
        exception = 0
        if method == 'beam_3' or method == 'beam_5':
            beam_k = int(method.split('_')[1])
        else:
            beam_k = 1

        total_score = dict()
        for metric in METRIC_LIST:
            total_score[metric] = 0

        for index, test in enumerate(tqdm(test_inputs)):
            predict, exception = integrated_prediction(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, beam_k, method, exception)
            for metric in METRIC_LIST:
                score = integrated_score(metric, test_outputs[index], predict)
                total_score[metric] += score

        for metric in METRIC_LIST:
            total_score[metric] = total_score[metric] / len(test_inputs)
            f_parameter.write('    '+metric+"="+str(round(total_score[metric], 4))+"\n")
            f_parameter.flush()
            print(metric+"="+str(round(total_score[metric], 4)))
                
        print("number of exception: ", exception, '\n')

    f_parameter.close()
