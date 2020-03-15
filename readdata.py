from util import *
from param import *
from tqdm import tqdm

'''
TODO:
    Change the rare words in comments into other common words via pre-trained embedding
'''

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

def countCodeToken(inputs: list):
    token_count = dict()
    # count the code tokens
    for index, pair in enumerate(tqdm(inputs)):
        pair = json.loads(pair)
        parsed_inputs = code_tokenize(pair['code'])
        inputs[index] = parsed_inputs
        if len(parsed_inputs) == 0:  # error-handling due to dirty data when SBT mode
            continue
        
        for x in parsed_inputs:
            if x not in token_count:
                token_count[x] = 1
            else:
                token_count[x] += 1
    return token_count

def SBT_OOVhandler(token_count: list, code_voc: list):
    code_tokens = []
    # select most frequency 30000 voc
    for w in sorted(token_count, key=token_count.get, reverse=True)[:30000-len(code_voc)]:
        code_voc.append(w) 
        
    # <SimpleName>_extractFor -> <SimpleName>, if <SimpleName>_extractFor is outside 30000 voc
    typename = ['<modifiers>', '<member>', '<value>', '<name>', '<operator>', '<qualifier>']
    for index, parsed_inputs in enumerate(tqdm(inputs)):
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

def extractSBTCode(inputs : list):
    code_voc = ['<PAD>','<START>','<END>','<UNK>','<modifiers>', '<member>', '<value>', '<name>', '<operator>', '<qualifier>']
    token_count = countCodeToken(inputs)
    code_voc, code_tokens = SBT_OOVhandler(token_count, code_voc)
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


if __name__ == '__main__':
    input_file = open('./simplified_dataset/simplified_train.json')
    inputs = input_file.readlines()
    start = time.time()
    print("comment tokenizing...")
    comment_voc, comment_tokens = extractComment()

    print("code tokenizing...")
    if MODE=="SBT":
        code_voc, code_tokens = extractSBTCode(inputs)

    elif MODE == "tok" or MODE == "symtok":
        code_voc, code_tokens = extractCode(inputs)

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
    if MODE=="symtok":
        pkl_filename = "./simplified_dataset/train_symtok_data.pkl"
    elif MODE=="tok":
        pkl_filename = "./simplified_dataset/train_tok_data.pkl"
    elif MODE=="SBT":
        pkl_filename = "./simplified_dataset/train_SBT_data.pkl"
        
    with open(pkl_filename, 'wb') as f:
        pickle.dump([code_train, comment_train, code_voc, comment_voc], f)

    print('size of code vocabulary: ', len(code_voc))
    print('size of comment vocabulary: ', len(comment_voc))