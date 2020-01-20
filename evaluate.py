from util import *
from model import *
from param import *
from predict import read_model, read_testset


BLEU_N = 3    # 3 (bleu3), 4 (bleu4)
PREDICT_METHOD = 1    # 0 (greedy search), 1 (beam search)
BEAM_SEARCH_K = 3    # 3 or 5


#  bleu4 (n=4)
def bleu(true, pred, n):
    true = nltk.word_tokenize(true)
    pred = nltk.word_tokenize(pred)
    c = len(pred)
    r = len(true)
    bp = 1. if c > r else np.exp(1 - r / (c + 1e-10))
    score = 0
    
    for i in range(1, n+1):
        true_ngram = set(ngram(true, i))
        pred_ngram = ngram(pred, i)
        if len(true_ngram)==0 or len(true_ngram)==0:
            break
        length = float(len(pred_ngram)) + 1e-10
        count = sum([1. if t in true_ngram else 0. for t in pred_ngram])
        score += math.log(1e-10 + (count / length))
    score = math.exp(score / n)  #n就是公式的Wn
    bleu = bp * score
    return bleu


if __name__ == '__main__':
    print("Reading "+MODE+" model...")
    code_voc, comment_voc, vocab_inp_size, vocab_tar_size, max_length_inp, max_length_targ = read_pkl()
    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    encoder, decoder = read_model(encoder, decoder)
    test_inputs, test_outputs = read_testset('./simplified_dataset/simplified_test.json')

    total_bleu = 0
    for index, test in enumerate(test_inputs):
        if PREDICT_METHOD == 0:
            predict = translate(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
        elif PREDICT_METHOD == 1:
            predict = ''
            try:
                predict = beam_search(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, BEAM_SEARCH_K)
            except:
                print('except')
        bleu_score = bleu(test_outputs[index], predict, BLEU_N)
        total_bleu += bleu_score
        if (index%2000) == 0:
            print(index)
            
    total_bleu = total_bleu / len(test_inputs)

    if MODE=="normal":
        checkpoint_dir = './training_checkpoints/adam-normal-256'
    elif MODE=="simple":
        checkpoint_dir = './training_checkpoints/adam-simple-256'
    elif MODE=="SBT":
        checkpoint_dir = './training_checkpoints/adam-SBT-256'

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