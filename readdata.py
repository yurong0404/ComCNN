from util import *
from param import *
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

def extractComment():
    comment_voc = ['<PAD>','<START>','<END>','<UNK>']
    comment_tokens = []
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        tokens = nltk.word_tokenize(pair['nl'])
        tokens.append('<END>')
        comment_tokens.append(tokens)
        for x in tokens:
            if x not in comment_voc:
                comment_voc.append(x)
    return comment_voc, comment_tokens


def extractCommentRemoveRareWord():
    comment_voc = ['<PAD>','<START>','<END>','<UNK>']
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
        if len(parsed_inputs) == 0:  # error-handling due to dirty data when SBT mode
            continue
        
        for x in parsed_inputs:
            if x not in token_count:
                token_count[x] = 1
            else:
                token_count[x] += 1
    return token_count

def extractSBTCode(inputs : list):
    code_voc = ['<PAD>','<START>','<END>','<UNK>','<modifiers>', '<member>', '<value>', '<name>', '<operator>', '<qualifier>']
    token_count = countCodeToken(inputs)
    code_voc.extend(sorted(token_count, key=token_count.get, reverse=True)[:30000-len(code_voc)])

    code_tokens = []
    # <SimpleName>_extractFor -> <SimpleName>, if <SimpleName>_extractFor is outside 30000 voc
    typename = ['<modifiers>', '<member>', '<value>', '<name>', '<operator>', '<qualifier>']
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        parsed_inputs = code_tokenize(pair['code'])
        if len(parsed_inputs) == 0:  
            continue
        for index2 in range(len(parsed_inputs)):
            if parsed_inputs[index2] not in code_voc:
                tmp = parsed_inputs[index2].split('_')
                if len(tmp) > 1 and tmp[0] in typename:
                    parsed_inputs[index2] = tmp[0]
                else:
                    parsed_inputs[index2] = "<UNK>"
        code_tokens.append(parsed_inputs)
    
    return code_voc, code_tokens


def extractCode(inputs: list):
    code_tokens = []
    code_voc = ['<PAD>','<START>','<END>','<UNK>']
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        parsed_inputs = code_tokenize(pair['code'])

        for x in parsed_inputs:
            if x not in code_voc:
                code_voc.append(x)
        code_tokens.append(parsed_inputs)
    return code_voc, code_tokens

def extractCodeRemoveRareWord(inputs: list):
    code_voc = ['<PAD>','<START>','<END>','<UNK>']
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


if __name__ == '__main__':
    input_file = open_trainset()
    inputs = input_file.readlines()
    start = time.time()
    print("comment tokenizing...")
    if MODE == "code2com" or MODE == "DeepCom":
        comment_voc, comment_tokens = extractCommentRemoveRareWord()
    else:
        comment_voc, comment_tokens = extractComment()

    print("code tokenizing...")
    if MODE == "SBT" or MODE == "DeepCom":
        code_voc, code_tokens = extractSBTCode(inputs)

    elif MODE == "tok":
        code_voc, code_tokens = extractCode(inputs)

    elif MODE == "CODE-NN" or MODE == "code2com":
        code_voc, code_tokens = extractCodeRemoveRareWord(inputs)

    input_file.close()

    print('readdata:')
    print('\tdata amount: '+str(len(code_tokens)))
    print('\trun time: '+str(time.time()-start))
    
    print('token2index...')
    code_train = token2index(code_tokens, code_voc)
    comment_train = token2index(comment_tokens, comment_voc)
    print('sequences padding...')
    code_train = pad_sequences(code_tokens, code_voc.index('<PAD>'))
    comment_train = pad_sequences(comment_tokens, comment_voc.index('<PAD>'))

    # Saving the training data:
    if MODE == "tok":
        pkl_filename = "./simplified_dataset/train_tok_data.pkl"
    elif MODE == "SBT":
        pkl_filename = "./simplified_dataset/train_SBT_data.pkl"
    elif MODE == "CODE-NN":
        pkl_filename = "./simplified_dataset/train_CODENN_data.pkl"
    elif MODE == "code2com":
        pkl_filename = "./simplified_dataset/train_code2com_data.pkl"
    elif MODE == "DeepCom":
        pkl_filename = "./simplified_dataset/train_DeepCom_data.pkl"

    with open(pkl_filename, 'wb') as f:
        pickle.dump([code_train, comment_train, code_voc, comment_voc], f)

    print('size of code vocabulary: ', len(code_voc))
    print('size of comment vocabulary: ', len(comment_voc))
