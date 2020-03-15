import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from util import *
from model import *
from param import *
from predict import read_model, read_testset
from tqdm import tqdm

METRIC = "BLEU3"   # BLEU3, BLEU4, CIDEr
PREDICT_METHOD = 0    # 0 (greedy search), 1 (beam search)
BEAM_SEARCH_K = 3    # 3 or 5
DATASET_LOC_LEVEL = 1  # 0 for 0~10, 1 for 10~20, 2 for 20~30, 3 for 30~40
DATASET_PATH = [
    './simplified_dataset/simplified_test_0_10.json',
    './simplified_dataset/simplified_test_10_20.json',
    './simplified_dataset/simplified_test_20_30.json',
    './simplified_dataset/simplified_test_30_40.json'
]


if __name__ == '__main__':
    print("metric:", METRIC)
    print("predict method:", PREDICT_METHOD)
    print("Reading "+MODE+" model...")
    code_train, comment_train, code_voc, comment_voc = read_pkl()
    vocab_inp_size = len(code_voc)
    vocab_tar_size = len(comment_voc)
    max_length_inp = max(len(t) for t in code_train)
    max_length_targ = max(len(t) for t in comment_train)
    if BIDIRECTIONAL==0:
        encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
        decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    elif BIDIRECTIONAL==1:
        encoder = BidirectionalEncoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
        decoder = BidirectionalDecoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    encoder, decoder = read_model(encoder, decoder)
    test_inputs, test_outputs = read_testset(DATASET_PATH[DATASET_LOC_LEVEL])

    total_score = 0
    exception = 0
    for index, test in enumerate(tqdm(test_inputs)):
        if PREDICT_METHOD==0 and BIDIRECTIONAL==0:
            predict = translate(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
        elif PREDICT_METHOD==0 and BIDIRECTIONAL==1:
            predict = translate_bilstm(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
        elif PREDICT_METHOD==1 and BIDIRECTIONAL==0:    
            predict = ''
            try:
                predict = beam_search(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, BEAM_SEARCH_K)
            except:
                exception += 1
        elif PREDICT_METHOD==1 and BIDIRECTIONAL==1:    
            predict = ''
            try:
                predict = beam_search_bilstm(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, BEAM_SEARCH_K)
            except:
                exception += 1
        
        if METRIC == "BLEU3":
            score = bleu(test_outputs[index], predict, 3)
        elif METRIC == "BLEU4":
            score = bleu(test_outputs[index], predict, 4)
        elif METRIC == "CIDEr":
            score = CIDEr(test_outputs[index], predict)
        total_score += score
            
    total_score = total_score / len(test_inputs)

    


    print(METRIC+":",round(total_score, 4))
    save_path = "./simplified_dataset/performance_"+str(DATASET_LOC_LEVEL*10)+"_"+str((DATASET_LOC_LEVEL+1)*10)
    if BIDIRECTIONAL == 0:
        model_arch = 'lstm'
    elif BIDIRECTIONAL == 1:
        model_arch = 'bilstm'
    if PREDICT_METHOD == 0:
        pred_method = 'greedy_search'
    elif PREDICT_METHOD == 1:
        pred_method = 'beam_search(k='+str(BEAM_SEARCH_K)+")"
    f_parameter = open(save_path, "a")
    f_parameter.write(METRIC+":\n")
    f_parameter.write(MODE+" "+model_arch+" "+pred_method+": "+str(round(total_score, 4))+"\n\n")
    f_parameter.close()

    print("number of exception: ", exception)
