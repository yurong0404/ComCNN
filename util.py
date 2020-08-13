import copy
import json
import math
import os
import pickle
import re
import javalang
import nltk
import numpy as np
import tensorflow as tf
from config import *
from model import *
from rouge_score import rouge_scorer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# nltk.download('punkt')


def split_identifier(identifier: str):
    """
    Usage:
    1. "camelCase" -> ["camel", "Case"]
    2. "snake_case" -> ["snake", "_", "case"]
    3. "normal" -> ["normal"]
    """
    if "_" in identifier:
        return identifier.split("_")
    elif identifier != identifier.lower() and identifier != identifier.upper():
        # regular expression for camelCase
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier
        )
        return [m.group(0) for m in matches]
    return [identifier]


def get_batch(inp: list, batch_sz: int):
    """
    Return shape:
        [None, batch_sz, None]
    Example:
        a = [1,2,3,4,5,6,7,8,9,10]
        a = getBatch(inp=a, batch_sz=3)
        ---output---
        [[1,2,3], [4,5,6], [7,8,9]]
    """
    dataset = []
    while len(inp) >= batch_sz:
        dataset.append(inp[:batch_sz])
        inp = inp[batch_sz:]
    if isinstance(inp, np.ndarray):
        return np.array(dataset)
    return dataset


def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))


def code_tokenize(code):
    inputs = []
    tokens_parse = javalang.tokenizer.tokenize(code)
    for token in tokens_parse:
        token = str(token).split(' ')
        # split the camelCase and snake_case
        splitted_id = split_identifier(token[1].strip('"'))
        inputs.extend(splitted_id)
    inputs.insert(0, '<START>')
    inputs.append('<END>')
    return inputs


def token_to_index(seq, voc):
    """
    ['public', 'void', ... '<END>'] -> [55, 66, ..., 2]
    """
    seq_index = []
    for token in seq:
        if token not in voc:
            seq_index.append(voc.index('<UNK>'))
        else:
            seq_index.append(voc.index(token))
    return seq_index


def token_zero_padding(seq, voc, max_length):
    # index of '<PAD>' is 0
    seq += [voc.index('<PAD>')] * (max_length - len(seq))
    seq = np.array(seq)
    return seq


def greedy_search(code, encoder, decoder, train_data):
    code = code_tokenize(code)
    if len(code) >= train_data['max_length_code']:
        code = code[:train_data['max_length_code']-1]
    code = token_to_index(code, train_data['code_voc'])
    code = token_zero_padding(code, train_data['code_voc'], train_data['max_length_code'])
    code = tf.expand_dims(code, 0)
    result = ''
    hidden = encoder.initialize_hidden_state(batch_sz=1)
    enc_output, enc_hidden_h, enc_hidden_c = encoder(code, hidden)
    dec_hidden = [enc_hidden_h, enc_hidden_c]
    dec_input = tf.expand_dims([train_data['comment_voc'].index('<START>')], 1)

    for t in range(train_data['max_length_com']):
        predictions, dec_hidden_h, dec_hidden_c = decoder(dec_input, dec_hidden, enc_output)
        dec_hidden = [dec_hidden_h, dec_hidden_c]
        predicted_id = tf.argmax(predictions[0]).numpy()
        if train_data['comment_voc'][predicted_id] == '<END>':
            return result
        result += train_data['comment_voc'][predicted_id] + ' '
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def predict_word_for_each_candidate(pred_info, decoder, enc_output, comment_voc, width):
    """
    it is called by beam search,
    it predicts one words for each of 'width * width' candidates,
    and also handle the joint scores, hidden states of decorder, etc.
    """
    cand_info = []
    for i in range(width):
        for x in range(width):
            cand_info += [copy.deepcopy(pred_info[i])]

        if pred_info[i]['end'] is True:
            continue

        predictions, dec_hidden_h, dec_hidden_c = decoder(
            pred_info[i]['dec_input'],
            pred_info[i]['dec_hidden'],
            enc_output
        )
        pred_info[i]['dec_hidden'] = [dec_hidden_h, dec_hidden_c]
        predictions = tf.nn.softmax(predictions)
        # pick out top k new words for each prediction comments
        topk_score = tf.math.top_k(predictions[0], width)[0]
        topk_id = tf.math.top_k(predictions[0], width)[1]

        for x in range(width):
            cand_info[width*i+x]['scores'] *= topk_score[x].numpy()
            if comment_voc[topk_id[x].numpy()] == '<END>':
                cand_info[width*i+x]['end'] = True
            else:
                cand_info[width*i+x]['gen_comments'] += comment_voc[topk_id[x].numpy()] + ' '
                cand_info[width*i+x]['dec_input'] = tf.expand_dims([topk_id[x].numpy()], 0)
                cand_info[width*i+x]['dec_hidden'] = pred_info[i]['dec_hidden']

    return cand_info, pred_info


def beam_search(code, encoder, decoder, train_data, width):
    code = code_tokenize(code)
    code = token_to_index(code, train_data['code_voc'])
    code = token_zero_padding(code, train_data['code_voc'], train_data['max_length_code'])
    code = tf.expand_dims(code, 0)

    hidden = encoder.initialize_hidden_state(batch_sz=1)
    enc_output, enc_hidden_h, enc_hidden_c = encoder(code, hidden)
    # pred_info : a list having 'width' elements, each element represent one prediction comment
    pred_info = []
    for i in range(width):
        pred_info.append({
            'gen_comments': '',
            'scores': 1,
            'end': False,  # used to determine whether the prediction comments end
            'dec_input': tf.expand_dims([train_data['comment_voc'].index('<START>')], 1),
            'dec_hidden': [enc_hidden_h, enc_hidden_c]
        })

    for t in range(train_data['max_length_com']):
        # cand_info : a list having 'width * width' elements
        cand_info, pred_info = predict_word_for_each_candidate(
            pred_info,
            decoder,
            enc_output,
            train_data['comment_voc'],
            width
        )
        # because the candidate of 1st iteration must be all the same
        if t == 0:
            pred_info = cand_info[:width]
            continue
        else:
            sorted_index = sorted(range(width ** 2), key=lambda k: cand_info[k]['scores'], reverse=True)[:width]
            # pick first 'width' best candidate comments as the temporary predictions
            for x in range(width):
                pred_info[x] = cand_info[sorted_index[x]]
            if False not in [pred_info[x]['end'] for x in range(width)]:
                break
    # select the comments with the highest joint scores as prediction
    return pred_info[0]['gen_comments']


def read_train_pkl():
    """
    return a dict(), having several training data information
    """
    f = open('./simplified_dataset/train_ComCNN_data.pkl', 'rb')
    code_train, comment_train, code_voc, comment_voc = pickle.load(f)
    code_voc_size = len(code_voc)
    com_voc_size = len(comment_voc)
    max_length_code = max(len(t) for t in code_train)
    max_length_com = max(len(t) for t in comment_train)
    train_data = {
        'code': code_train,
        'comment': comment_train,
        'code_voc': code_voc,
        'comment_voc': comment_voc,
        'code_voc_size': code_voc_size,
        'com_voc_size': com_voc_size,
        'max_length_code': max_length_code,
        'max_length_com': max_length_com
    }
    return train_data


def read_testset(**kwargs):
    """
    return a list of many dicts having 'code' and 'comment' keys
    """
    if "path" in kwargs:
        f = open(kwargs['path'])
    else:
        f = open('./simplified_dataset/simplified_test.json')
    inputs = f.readlines()
    f.close()
    test_data = []

    for pair in inputs:
        pair = json.loads(pair)
        test_data.append({'code': pair['code'], 'comment': pair['nl']})

    return test_data


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
        if len(true_ngram) == 0 or len(pred_ngram) == 0:
            break
        length = float(len(pred_ngram)) + 1e-10
        count = sum([1. if t in true_ngram else 0. for t in pred_ngram])
        score += math.log(1e-10 + (count / length))
    # n就是公式的Wn
    score = math.exp(score / n)
    bleu = bp * score
    return bleu


def tf_idf(ngram_list, ngram, total_ngram_count):
    count = ngram_list.count(ngram)
    tf = count / total_ngram_count
    # in our dataset, tf-idf is either (tf*1) or (0* every large number)
    # so idf=1 results in the same consequence
    idf = 1
    return tf * idf


def cider(true, pred):
    true = nltk.word_tokenize(true)
    pred = nltk.word_tokenize(pred)
    N = 4
    cider_score = 0
    for n in range(1, 5):
        true_ngram = ngram(true, n)
        pred_ngram = ngram(pred, n)
        if len(true_ngram) == 0 or len(pred_ngram) == 0:
            break

        total_ngram = true_ngram + pred_ngram
        total_ngram_count_in_cand = 1e-10
        total_ngram_count_in_ref = 1e-10

        for t in set(total_ngram):
            total_ngram_count_in_cand += pred_ngram.count(t)
            total_ngram_count_in_ref += true_ngram.count(t)
        g_cand = [tf_idf(pred_ngram, t, total_ngram_count_in_cand) for t in set(total_ngram)]
        g_ref = [tf_idf(true_ngram, t, total_ngram_count_in_ref) for t in set(total_ngram)]

        # inner product of two list
        g = sum([a*b for a, b in zip(g_cand, g_ref)])
        abs_cand = sum([a**2 for a in g_cand]) ** 0.5
        abs_ref = sum([a**2 for a in g_ref]) ** 0.5
        cider_score += (g / (abs_cand * abs_ref)) / N
    return cider_score


def get_checkpoint_dir():
    checkpoint_dir = ''
    if ARCH == "lstm_lstm":
        checkpoint_dir = './training_checkpoints/ComCNN-lstm-lstm'
    elif ARCH == "cnnlstm_lstm":
        checkpoint_dir = './training_checkpoints/ComCNN-cnnlstm-lstm'
    elif ARCH == "cnnbilstm_lstm":
        checkpoint_dir = './training_checkpoints/ComCNN-cnnbilstm-lstm'
    else:
        print('Error: get_checkpoint_dir')
        exit(0)
    return checkpoint_dir


def restore_model(encoder, decoder):
    checkpoint_dir = get_checkpoint_dir()
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    return encoder, decoder


def integrated_prediction(code, encoder, decoder, train_data, beam_k, method):
    if method == 'greedy':
        predict = greedy_search(code, encoder, decoder, train_data)
    elif method == 'beam_3' or method == 'beam_5':
        predict = beam_search(code, encoder, decoder, train_data, beam_k)
    return predict


def integrated_score(metric, test_output, predict):
    score = 0
    if metric == 'BLEU3':
        score = bleu(test_output, predict, 3)
    elif metric == 'BLEU4':
        score = bleu(test_output, predict, 4)
    elif metric == 'CIDEr':
        score = cider(test_output, predict)
    elif metric == 'ROUGE_L':
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        score = scorer.score(test_output, predict)['rougeL'].fmeasure
    return score


def create_model(vocab_inp_size, vocab_tar_size, max_length_inp):
    if ARCH == "lstm_lstm":
        encoder = lstmEncoder(vocab_inp_size, EMBEDDING_DIM, UNITS)
        decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS)
    elif ARCH == "cnnlstm_lstm":
        encoder = cnnlstmEncoder(vocab_inp_size, EMBEDDING_DIM, UNITS, max_length_inp)
        decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS)
    elif ARCH == "cnnbilstm_lstm":
        encoder = cnnbilstmEncoder(vocab_inp_size, EMBEDDING_DIM, FILTERS, max_length_inp)
        decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, FILTERS)

    return encoder, decoder
