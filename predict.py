import os
from util import *
from model import *
from config import *

GREEN = "\033[0;32;40m"
RED = "\033[0;31;40m"
RESET_COLOR = "\033[0m"

if __name__ == '__main__':
    print("Reading "+MODE+" model...")
    code_train, comment_train, code_voc, comment_voc = read_pkl()
    vocab_inp_size = len(code_voc)
    vocab_tar_size = len(comment_voc)
    max_length_inp = max(len(t) for t in code_train)
    max_length_targ = max(len(t) for t in comment_train)
    
    encoder, decoder= create_encoder_decoder(vocab_inp_size, vocab_tar_size, max_length_inp)
    
    encoder, decoder = read_model(encoder, decoder)
    test_inputs, test_outputs = read_testset('./simplified_dataset/simplified_test.json')

    input_type = 0 # 0 for testing set, 1 for custom
    input_type = int(input("choose the type of input, 0 (testing set) / 1 (custom): "))
    if input_type != 0 and input_type!=1:
        print('wrong input')
        exit(0)
    while(1):
        print(RED+"\n"+"="*80+RESET_COLOR)
        if input_type == 0:
            index = int(input("code number: "))
            if index == -1:
                break
            print(GREEN+"\nCode:"+RESET_COLOR)
            code = test_inputs[index]
            print(code)
            print(GREEN+"Original comment:\n"+RESET_COLOR+test_outputs[index])
            predict = translate(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
            print(GREEN+"\nGreedy search prediction:\n"+RESET_COLOR+ predict)
            predict = beam_search(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, 3)
            print(GREEN+"\nBeam search prediction (k=3):\n"+RESET_COLOR+ predict)
            predict = beam_search(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, 5)
            print(GREEN+"\nBeam search prediction (k=5):\n"+RESET_COLOR+ predict)
        elif input_type == 1:
            print(GREEN+"\nCode:"+RESET_COLOR)
            lines = []
            while True:
                line = input()
                if line:
                    lines.append(line)
                else:
                    break
            code = '\n'.join(lines)
            predict = translate(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
            print(GREEN+"\nGreedy search prediction:\n"+RESET_COLOR+ predict)
            predict = beam_search(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, 3)
            print(GREEN+"\nBeam search prediction (k=3):\n"+RESET_COLOR+ predict)
            predict = beam_search(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, 5)
            print(GREEN+"\nBeam search prediction (k=5):\n"+RESET_COLOR+ predict)
                
