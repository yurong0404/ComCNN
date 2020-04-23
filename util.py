import javalang
import json
import re
import time
import nltk
#nltk.download('punkt')
import numpy as np
import pickle
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from param import *
from model import *
from rouge_score import rouge_scorer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    
'''
Function: 
    Input the root of AST and the deep of the tree, 
    it will filter the null value and return the list of SBT (structural-based travesal) and print the tree structure
'''
def parse_tree(root, deep):
    seq = []
    seq.extend(['(', str(root).split('(')[0]])
    #print('\t'*(deep)+str(root).split('(')[0])    # show node name
    if not hasattr(root, 'attrs'):  # error-handling
        return []
    for attr in root.attrs:
        if eval('root.%s' % attr) in [None, [], "", set(), False]:    # filter the null attr
            continue
        elif isinstance(eval('root.%s' % attr), list):
            x = eval('root.%s' % attr)
            if not all(elem in x for elem in [None, [], "", set(), False]):    # if not all elements in list are null
                seq.extend(['(',attr])
                #print('\t'*(deep+1)+attr)
                #deep += 1
                for i in eval('root.%s' % attr):    # recursive the list
                    if i is None or isinstance(i, str):    # perhaps it has None value in the list
                        continue
                    #deep += 1
                    seq.extend(parse_tree(i, deep))
                    
                    #deep -= 1
                #deep -= 1
                seq.extend([')',attr])
        elif 'tree' in str(type(eval('root.%s' % attr))):    #if the attr is one kind of Node, recursive the Node
            seq.extend(['(',attr])
            #print('\t'*(deep+1)+attr)
            #deep += 2
            seq.extend(parse_tree(eval('root.%s' % attr), deep))
            #deep -= 2
            seq.extend([')',attr])
        else:
            seq.extend(['(','<'+str(attr)+'>_'+str(eval('root.%s' % attr)),')','<'+str(attr)+'>_'+str(eval('root.%s' % attr))])
            #exec("print('\t'*(deep+1)+attr+': '+str(root.%s))" % attr)    #it must be normal attribute
    seq.extend([')', str(root).split('(')[0]])
    return seq


'''
Usage:
    1. "camelCase" -> ["camel", "Case"]
    2. "snake_case" -> ["snake", "_", "case"]
    3. "normal" -> ["normal"]
'''
def split_identifier(id_token: str):
    if  "_" in id_token:
        return id_token.split("_")
    elif id_token != id_token.lower() and id_token != id_token.upper():
        # regular expression for camelCase
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', id_token)
        return [m.group(0) for m in matches]
    else:
        return [id_token]



'''
Usage:
    Transform the token to the index in vocabulary
    ['<START>', '<Modifier>', 'public', ..., '<Separator>', ';', '<Separator>', '}', '<END>']
    => [0, 7, 8, ..., 14, 29, 14, 30, 1]
'''
def token2index(lst: list, voc: list) -> list:
    for index, seq in enumerate(lst):
        seq_index = []
        for token in seq:
            seq_index.append(voc.index(token))
        lst[index] = seq_index
    return lst


def pad_sequences(lst: list, pad_data: int):
    maxlen = max(len(x) for x in lst)
    for index, seq in enumerate(lst):
        lst[index].extend([pad_data] * (maxlen-len(seq)))
    return np.array(lst)

'''
Return shape:
    [None, batch_sz, None]
Example:
    a = [1,2,3,4,5,6,7,8,9,10]
    a = getBatch(x=a, batch_sz=3)
    a
    ---output---
    [[1,2,3], [4,5,6], [7,8,9]]
'''
def getBatch(x: list, batch_sz: int):
    dataset = []
    while(len(x)>=batch_sz):
        dataset.append(x[:batch_sz])
        x = x[batch_sz:]
    if type(x) == np.ndarray:
        return np.array(dataset)
    elif type(x) == list:
        return dataset
    
def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))


def code_to_index(inputs, code_voc, max_length_inp):
    if len(inputs) >= max_length_inp:
        inputs = inputs[:max_length_inp-1]

    if MODE=="CODE-NN" or MODE == "ComCNN":
        for index, token in enumerate(inputs):
            if token not in code_voc:
                inputs[index] = code_voc.index('<UNK>')
            else:
                inputs[index] = code_voc.index(token)
                
    elif MODE=="DeepCom":
        typename = ['<modifiers>', '<member>', '<value>', '<name>', '<operator>', '<qualifier>']
        for index, token in enumerate(inputs):
            if token not in code_voc:
                tmp = token.split('_')
                if len(tmp) > 1 and tmp[0] in typename:
                    inputs[index] = code_voc.index(tmp[0])
                else:
                    inputs[index] = code_voc.index("<UNK>")
            else:
                inputs[index] = code_voc.index(token)
                
    inputs += [code_voc.index('<PAD>')] * (max_length_inp - len(inputs))
    inputs = np.array(inputs)
    inputs = tf.expand_dims(inputs, 0)

    return inputs


def code_tokenize(code):
    inputs = []
    if MODE == "ComCNN":
        tokens_parse = javalang.tokenizer.tokenize(code)
        for token in tokens_parse:    # iterate the tokens of the sentence
            token = str(token).split(' ')
            splitted_id = split_identifier(token[1].strip('"'))    # split the camelCase and snake_case
            inputs.extend(splitted_id)
            
    elif MODE == "DeepCom":
        tree = javalang.parse.parse('class aa {'+code+'}')
        _, node = list(tree)[2]    # 前兩個用來篩掉class aa{ }的部分
        inputs = parse_tree(node, 0)
        if len(inputs) == 0:   # error-handling due to dirty data
            return []
    
    elif MODE == "CODE-NN":
        tokens_parse = javalang.tokenizer.tokenize(code)
        for token in tokens_parse:
            token = str(token).split(' ')
            token[1] = token[1].strip('"')
            inputs.append(token[1])

    inputs.insert(0, '<START>')
    inputs.append('<END>')
    
    return inputs


'''
用途：把一個二維的array做機率正規化
例如：[[3,4,5],[1,2,3]] -> [[0.25, 0.33, 0.416], [0.167, 0.333, 0.5]]
'''
def distribution(arr):
    new_arr = []
    for i in arr:
        tmp = []
        total = sum(i)
        for x in i:
            tmp.append(x/total)
        new_arr.append(tmp)
    return np.array(new_arr)


def enc_output_init_dec_hidden(inputs, encoder, decoder):
    if ARCH == "lstm":
        hidden_h, hidden_c = tf.zeros((1, encoder.enc_units)), tf.zeros((1, encoder.enc_units))
        hidden = [hidden_h, hidden_c]
        enc_output, enc_hidden_h, enc_hidden_c = encoder(inputs, hidden)
        dec_hidden = [enc_hidden_h, enc_hidden_c]
    elif ARCH == "bilstm":
        hidden = [tf.zeros((1, encoder.enc_units)), tf.zeros((1, encoder.enc_units)), \
                    tf.zeros((1, encoder.enc_units)), tf.zeros((1, encoder.enc_units))]
        enc_output, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = encoder(inputs, hidden)
        dec_hidden = [enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c]
    elif ARCH == "cnn_lstm":
        enc_output = encoder(inputs)
        dec_hidden = [tf.zeros((1, decoder.dec_units)), tf.zeros((1, decoder.dec_units))]
    elif ARCH == "cnn_bilstm":
        enc_output = encoder(inputs)
        dec_hidden = [tf.zeros((1, decoder.dec_units)), tf.zeros((1, decoder.dec_units)), \
                        tf.zeros((1, decoder.dec_units)), tf.zeros((1, decoder.dec_units))]

    elif ARCH == "CODE-NN":
        hidden_h, hidden_c = tf.zeros((1, encoder.enc_units)), tf.zeros((1, encoder.enc_units))
        hidden = [hidden_h, hidden_c]
        enc_output, enc_hidden_h, enc_hidden_c = encoder(inputs, hidden)
        dec_hidden = [enc_hidden_h, enc_hidden_c]
    return enc_output, dec_hidden

def decode_iterate(decoder, dec_input, dec_hidden, enc_output, code):
    if ARCH == "lstm" or ARCH == "cnn_lstm":
        predictions, dec_hidden_h, dec_hidden_c = decoder(dec_input, dec_hidden, enc_output)
        dec_hidden = [dec_hidden_h, dec_hidden_c]
    elif ARCH == "bilstm" or ARCH == "cnn_bilstm":
        predictions, dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c = decoder(dec_input, dec_hidden, enc_output)
        dec_hidden = [dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c]
    elif ARCH == "CODE-NN":
        predictions, dec_hidden_h, dec_hidden_c = decoder(dec_input, dec_hidden, code)
        dec_hidden = [dec_hidden_h, dec_hidden_c]
    return predictions, dec_hidden
    
def translate(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ):
    
    inputs = code_tokenize(code)
    inputs = code_to_index(inputs, code_voc, max_length_inp)
    result = ''
    enc_output, dec_hidden = enc_output_init_dec_hidden(inputs, encoder, decoder)
    dec_input = tf.expand_dims([comment_voc.index('<START>')], 1)       
    
    for t in range(max_length_targ):
        predictions, dec_hidden = decode_iterate(decoder, dec_input, dec_hidden, enc_output, inputs)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if comment_voc[predicted_id] == '<END>':
            return result
        result += comment_voc[predicted_id] + ' '
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def beam_search_predict_word(lock, score, result, decoder, dec_input, dec_hidden, enc_output, comment_voc, code, width):
    can_lock = [0] * (width ** 2)
    can_input = [''] * (width ** 2)
    can_score = [1] * (width ** 2)
    can_result = [''] * (width ** 2)

    for i in range(width):
        for x in range(width):
            can_score[width*i+x] = score[i]
            can_result[width*i+x] = result[i]
        if lock[i] == 1:
            for x in range(width):
                can_lock[width*i+x] = 1
            continue
        
        predictions, dec_hidden[i] = decode_iterate(decoder, dec_input[i], dec_hidden[i], enc_output, code)
            
        predictions = tf.nn.softmax(predictions)
        topk_score = tf.math.top_k(predictions[0], width)[0]
        topk_id = tf.math.top_k(predictions[0], width)[1]
        
        for x in range(width):
            can_score[width*i+x] *= topk_score[x].numpy()
            if comment_voc[topk_id[x].numpy()] == '<END>':
                can_lock[width*i+x] = 1
            else:
                can_result[width*i+x] += comment_voc[topk_id[x].numpy()] + ' '
                can_input[width*i+x] = topk_id[x].numpy()
    return can_lock, can_score, can_result, can_input, dec_hidden

def beam_search_generate_topk_candidate(can_score, can_result, can_lock, can_input, result, score, lock, dec_hidden, dec_input, width):
    sorted_index = sorted(range(len(can_score)), key=lambda k: can_score[k], reverse=True)[:width]
    for x in range(width):
        result[x] = can_result[sorted_index[x]]
        score[x] = can_score[sorted_index[x]]
        if can_lock[sorted_index[x]] == 1:
            lock[x] = 1
        else:
            dec_input[x] = tf.expand_dims([can_input[sorted_index[x]]], 0)
        dec_hidden[x] = dec_hidden[sorted_index[x]//width]
    return lock, result, score, dec_input, dec_hidden


def beam_search(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, width):
    inputs = code_tokenize(code)
    inputs = code_to_index(inputs, code_voc, max_length_inp)

    enc_output, dec_hidden = enc_output_init_dec_hidden(inputs, encoder, decoder)
    dec_hidden = [dec_hidden] * width

    dec_input = [tf.expand_dims([comment_voc.index('<START>')], 1)] * width
    
    result = [''] * width
    score = [1] * width
    lock = [0] * width
    
    for t in range(max_length_targ):
        can_lock, can_score, can_result, can_input, dec_hidden = beam_search_predict_word(lock, score, result, decoder, dec_input, dec_hidden, enc_output, comment_voc, inputs, width)

        if t == 0:
            result[:width] = can_result[:width]
            score[:width] = can_score[:width]
            dec_input = [tf.expand_dims([can_input[x]], 0) for x in range(width)]
            continue
        
        lock, result, score, dec_input, dec_hidden = beam_search_generate_topk_candidate(can_score, can_result, can_lock, can_input, result, score, lock, dec_hidden, dec_input, width)
        if 0 not in lock:
            break
    return result[0]


# Read the training data:
def read_pkl():
    if MODE=="CODE-NN":
        f = open('./simplified_dataset/train_CODENN_data.pkl', 'rb')
    elif MODE=="ComCNN":
        f = open('./simplified_dataset/train_ComCNN_data.pkl', 'rb')
    elif MODE=="DeepCom":
        f = open('./simplified_dataset/train_DeepCom_data.pkl', 'rb')
    code_train, comment_train, code_voc, comment_voc = pickle.load(f)
    
    return code_train, comment_train, code_voc, comment_voc

def read_testset(path):
    f = open(path)
    inputs = f.readlines()
    f.close()
    test_inputs = []
    test_outputs = []

    for pair in inputs:
        pair = json.loads(pair)
        test_inputs.append(pair['code'])
        test_outputs.append(pair['nl'])
    
    return test_inputs, test_outputs

def open_trainset():
    f = open('./simplified_dataset/simplified_train.json')
    return f


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
        if len(true_ngram)==0 or len(pred_ngram)==0:
            break
        length = float(len(pred_ngram)) + 1e-10
        count = sum([1. if t in true_ngram else 0. for t in pred_ngram])
        score += math.log(1e-10 + (count / length))
    score = math.exp(score / n)  #n就是公式的Wn
    bleu = bp * score
    return bleu


def TF_IDF(ngram_list, ngram, total_ngram_count):
    count = ngram_list.count(ngram)
    tf = count / total_ngram_count
    # in the case of our dataset, tf-idf is either (tf*1) or (0* every large number)
    # so idf=1 results in the same consequence
    idf = 1
    return tf * idf

def CIDEr(true, pred):
    true = nltk.word_tokenize(true)
    pred = nltk.word_tokenize(pred)
    N = 4
    CIDEr_score = 0
    for n in range(1,5):
        true_ngram = ngram(true, n)
        pred_ngram = ngram(pred, n)
        if len(true_ngram)==0 or len(pred_ngram)==0:
            break
        
        total_ngram = true_ngram + pred_ngram
        total_ngram_count_in_cand = 1e-10
        total_ngram_count_in_ref = 1e-10

        for t in set(total_ngram):
            total_ngram_count_in_cand += pred_ngram.count(t)
            total_ngram_count_in_ref += true_ngram.count(t)
        g_cand = [TF_IDF(pred_ngram, t, total_ngram_count_in_cand) for t in set(total_ngram)]
        g_ref = [TF_IDF(true_ngram, t, total_ngram_count_in_ref) for t in set(total_ngram)]

        # inner product of two list
        g = sum([a*b for a,b in zip(g_cand, g_ref)])
        abs_cand = sum([a**2 for a in g_cand]) ** 0.5
        abs_ref = sum([a**2 for a in g_ref]) ** 0.5
        CIDEr_score += (g / (abs_cand * abs_ref)) / N
    return CIDEr_score
        

def getCheckpointDir():
    checkpoint_dir = ''
    if MODE=="CODE-NN" and ARCH=="CODE-NN":
        checkpoint_dir = './training_checkpoints/CODENN'
    elif MODE=="ComCNN" and ARCH=="lstm":
        checkpoint_dir = './training_checkpoints/ComCNN-lstm'
    elif MODE=="ComCNN" and ARCH=="cnn_lstm":
        checkpoint_dir = './training_checkpoints/ComCNN-cnn'
    elif MODE=="ComCNN" and ARCH=="cnn_bilstm":
        checkpoint_dir = './training_checkpoints/ComCNN-cnnbilstm'
    elif MODE=="DeepCom" and ARCH=="lstm":
        checkpoint_dir = './training_checkpoints/DeepCom-lstm'
    else:
        print('Error: getCheckpointDir')
        exit(0)
    return checkpoint_dir

def read_model(encoder, decoder):
    checkpoint_dir = getCheckpointDir()
    
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    return encoder, decoder

def integrated_prediction(test_input, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, beam_k, method, exception):
    if method == 'greedy':
        predict = translate(test_input, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
    elif method=='beam_3' or method=='beam_5':
        predict = ''
        try:
            predict = beam_search(test_input, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, beam_k)
        except:
            exception += 1
    return predict, exception


def integrated_score(metric, test_output, predict):
    score = 0
    if metric == 'BLEU3':
        score = bleu(test_output, predict, 3)
    elif metric == 'BLEU4':
        score = bleu(test_output, predict, 4)
    elif metric == 'CIDEr':
        score = CIDEr(test_output, predict)
    elif metric == 'ROUGE_L':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        score = scorer.score(test_output, predict)['rougeL'].fmeasure
    return score

def create_encoder_decoder(vocab_inp_size, vocab_tar_size, max_length_inp):
    if ARCH == "lstm":
        encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
        decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    elif ARCH == "bilstm":
        encoder = BidirectionalEncoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
        decoder = BidirectionalDecoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    elif ARCH == "cnn_lstm":
        encoder = cnnEncoder(vocab_inp_size, EMBEDDING_DIM, FILTERS, BATCH_SIZE, max_length_inp)
        decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, FILTERS, BATCH_SIZE)
    elif ARCH == "cnn_bilstm":
        encoder = cnnEncoder(vocab_inp_size, EMBEDDING_DIM, FILTERS, BATCH_SIZE, max_length_inp)
        decoder = BidirectionalDecoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    elif ARCH == "CODE-NN":
        encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, FILTERS, BATCH_SIZE)
        decoder = codennDecoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE, vocab_inp_size)

    return encoder, decoder
