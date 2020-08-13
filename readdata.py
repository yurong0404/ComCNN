import time
from util import *
from config import *
from tqdm import tqdm


def countCommentToken(inputs: list):
    token_count = dict()
    # count the comment tokens
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        tokens = nltk.word_tokenize(pair['nl'])

        for x in tokens:
            if x not in token_count:
                token_count[x] = 1
            else:
                token_count[x] += 1
    return token_count


def extractComment(inputs):
    comment_voc = ['<PAD>', '<START>', '<END>', '<UNK>']
    token_count = countCommentToken(inputs)

    keys = list(token_count.keys())
    for i in keys:
        if token_count[i] < 3:
            del token_count[i]
    comment_voc.extend(list(token_count.keys()))

    comment_tokens = []
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        tokens = nltk.word_tokenize(pair['nl'])
        tokens.append('<END>')
        for index2 in range(len(tokens)):
            if tokens[index2] not in comment_voc:
                tokens[index2] = "<UNK>"
        comment_tokens.append(tokens)

    return comment_voc, comment_tokens


def countCodeToken(inputs: list):
    token_count = dict()
    # count the code tokens
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        parsed_inputs = code_tokenize(pair['code'])
        for x in parsed_inputs:
            if x not in token_count:
                token_count[x] = 1
            else:
                token_count[x] += 1
    return token_count


def extractCodeRemoveRare(inputs: list):
    code_voc = ['<PAD>', '<START>', '<END>', '<UNK>']
    token_count = countCodeToken(inputs)
    keys = list(token_count.keys())
    for i in keys:
        if token_count[i] < 3:
            del token_count[i]
    code_voc.extend(list(token_count.keys()))

    code_tokens = []
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        parsed_inputs = code_tokenize(pair['code'])
        for index2 in range(len(parsed_inputs)):
            if parsed_inputs[index2] not in code_voc:
                parsed_inputs[index2] = "<UNK>"
        code_tokens.append(parsed_inputs)

    return code_voc, code_tokens


def extractCode(inputs: list):
    code_voc = ['<PAD>', '<START>', '<END>', '<UNK>']
    code_tokens = []
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        parsed_inputs = code_tokenize(pair['code'])
        for token in parsed_inputs:
            if token not in code_voc:
                code_voc.append(token)
        code_tokens.append(parsed_inputs)

    return code_voc, code_tokens


if __name__ == '__main__':
    input_file = open('./simplified_dataset/simplified_train.json')
    inputs = input_file.readlines()
    start = time.time()
    print("comment tokenizing...")
    comment_voc, comment_tokens = extractComment(inputs)

    print("code tokenizing...")
    code_voc, code_tokens = extractCode(inputs)

    input_file.close()

    print('token to index...')
    for index in tqdm(range(len(code_tokens))):
        code_tokens[index] = token_to_index(code_tokens[index], code_voc)
    for index in tqdm(range(len(comment_tokens))):
        comment_tokens[index] = token_to_index(comment_tokens[index], comment_voc)

    print('sequences padding...')
    max_length_code = max(len(x) for x in code_tokens)
    for index in tqdm(range(len(code_tokens))):
        code_tokens[index] = token_zero_padding(code_tokens[index], code_voc, max_length_code)
    code_tokens = np.array(code_tokens)
    max_length_com = max(len(x) for x in comment_tokens)
    for index in tqdm(range(len(comment_tokens))):
        comment_tokens[index] = token_zero_padding(comment_tokens[index], comment_voc, max_length_com)
    comment_tokens = np.array(comment_tokens)

    print('readdata:')
    print('\tdata amount: '+str(len(code_tokens)))
    print('\trun time: '+str(time.time()-start))

    # Saving the training data:
    pkl_filename = "./simplified_dataset/train_ComCNN_data.pkl"

    with open(pkl_filename, 'wb') as f:
        pickle.dump([code_tokens, comment_tokens, code_voc, comment_voc], f)

    print('size of code vocabulary: ', len(code_voc))
    print('size of comment vocabulary: ', len(comment_voc))
