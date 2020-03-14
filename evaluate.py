import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from util import *
from model import *
from param import *
from predict import read_model, read_testset


BLEU_N = 4    # 3 (bleu3), 4 (bleu4)
PREDICT_METHOD = 1    # 0 (greedy search), 1 (beam search)
BEAM_SEARCH_K = 5    # 3 or 5


if __name__ == '__main__':
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
    test_inputs, test_outputs = read_testset('./simplified_dataset/simplified_test.json')

    total_bleu = 0
    for index, test in enumerate(test_inputs):
        if PREDICT_METHOD==0 and BIDIRECTIONAL==0:
            predict = translate(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
        elif PREDICT_METHOD==0 and BIDIRECTIONAL==1:
            predict = translate_bilstm(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
        elif PREDICT_METHOD==1 and BIDIRECTIONAL==0:    
            predict = ''
            try:
                predict = beam_search(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, BEAM_SEARCH_K)
            except:
                print('except')
        elif PREDICT_METHOD==1 and BIDIRECTIONAL==1:    
            predict = ''
            try:
                predict = beam_search_bilstm(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, BEAM_SEARCH_K)
            except:
                print('except')
        bleu_score = bleu(test_outputs[index], predict, BLEU_N)
        total_bleu += bleu_score
        if (index%2000) == 0:
            print(index)
            
    total_bleu = total_bleu / len(test_inputs)

    checkpoint_dir = getCheckpointDir()

    if PREDICT_METHOD == 0:
        print("bleu"+str(BLEU_N)+":",round(total_bleu, 4))
        f_parameter = open(checkpoint_dir+"/parameters", "a")
        f_parameter.write("BLEU"+str(BLEU_N)+"="+str(round(total_bleu, 4))+"\n")
        f_parameter.close()
    elif PREDICT_METHOD == 1:
        print("Beam search(k="+str(BEAM_SEARCH_K)+") bleu"+str(BLEU_N)+":",round(total_bleu, 4))
        f_parameter = open(checkpoint_dir+"/parameters", "a")
        f_parameter.write("Beam search(k="+str(BEAM_SEARCH_K)+") BLEU"+str(BLEU_N)+"="+str(round(total_bleu, 4))+"\n")
        f_parameter.close()
