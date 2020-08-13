import os
from util import *
from model import *
from config import *

GREEN = "\033[0;32;40m"
RED = "\033[0;31;40m"
RESET_COLOR = "\033[0m"

if __name__ == '__main__':
    print("Reading "+ARCH+" model...")
    train_data = read_train_pkl()
    encoder, decoder = create_model(
        train_data['code_voc_size'],
        train_data['com_voc_size'],
        train_data['max_length_code']
    )
    encoder, decoder = restore_model(encoder, decoder)
    test_data = read_testset()

    input_type = 0  # 0 for testing set, 1 for custom
    input_type = int(input("choose the type of input, 0 (testing set) / 1 (custom): "))
    if input_type != 0 and input_type != 1:
        print('wrong input')
        exit(0)
    while(1):
        print(RED+"\n"+"="*80+RESET_COLOR)
        if input_type == 0:
            index = int(input("code number: "))
            if index == -1:
                break
            print(GREEN+"\nCode:"+RESET_COLOR)
            code = test_data[index]['code']
            print(code)
            print(GREEN+"Original comment:\n"+RESET_COLOR+test_data[index]['comment'])
            predict = greedy_search(code, encoder, decoder, train_data)
            print(GREEN + "\nGreedy search prediction:\n" + RESET_COLOR + predict)
            predict = beam_search(code, encoder, decoder, train_data, 3)
            print(GREEN + "\nBeam search prediction (k=3):\n" + RESET_COLOR + predict)
            predict = beam_search(code, encoder, decoder, train_data, 5)
            print(GREEN + "\nBeam search prediction (k=5):\n" + RESET_COLOR + predict)
            predict = beam_search(code, encoder, decoder, train_data, 7)
            print(GREEN + "\nBeam search prediction (k=7):\n" + RESET_COLOR + predict)
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
            # predict = translate(code, encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
            predict = greedy_search(code, encoder, decoder, train_data)
            print(GREEN + "\nGreedy search prediction:\n" + RESET_COLOR + predict)
            predict = beam_search(code, encoder, decoder, train_data, 3)
            print(GREEN + "\nBeam search prediction (k=3):\n" + RESET_COLOR + predict)
            predict = beam_search(code, encoder, decoder, train_data, 5)
            print(GREEN + "\nBeam search prediction (k=5):\n" + RESET_COLOR + predict)
            predict = beam_search(code, encoder, decoder, train_data, 7)
            print(GREEN + "\nBeam search prediction (k=7):\n" + RESET_COLOR + predict)
