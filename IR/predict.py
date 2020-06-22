import sys
sys.path.append('../')
from IR import lev_translate, token2index, index2token, read_testset
import pickle

GREEN = "\033[0;32;40m"
RED = "\033[0;31;40m"
RESET_COLOR = "\033[0m"

if __name__ == '__main__':
    f = open('../simplified_dataset/train_CODENN_data.pkl', 'rb')
    code_train, comment_train, code_voc, comment_voc = pickle.load(f)
    f.close()
    test_inputs, test_outputs = read_testset('../simplified_dataset/simplified_test.json')

    while(1):
        print(RED+"\n"+"="*80+RESET_COLOR)
        index = int(input("code number: "))
        if index == -1:
            break
        print(GREEN+"\nCode:"+RESET_COLOR)
        code = test_inputs[index]
        print(code)
        print(GREEN+"Original comment:\n"+RESET_COLOR+test_outputs[index])
        seq = token2index(code, code_voc)
        comment_index = lev_translate(seq, code_train)
        predict = comment_train[comment_index]
        str = ' '.join([i for i in index2token(predict, comment_voc) if i != "<PAD>"])
        print(GREEN, "\nIR prediction:\n", RESET_COLOR, str)
