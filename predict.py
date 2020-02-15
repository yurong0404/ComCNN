import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from util import *
from model import *
from param import *

def read_model(encoder, decoder):
    if MODE=="normal" and BIDIRECTIONAL==0:
        checkpoint_dir = './training_checkpoints/adam-normal-256'
    elif MODE=="simple" and BIDIRECTIONAL==0:
        checkpoint_dir = './training_checkpoints/adam-simple-256'
    elif MODE=="SBT" and BIDIRECTIONAL==0:
        checkpoint_dir = './training_checkpoints/adam-SBT-256'
    elif MODE=="normal" and BIDIRECTIONAL==1:
        checkpoint_dir = './training_checkpoints/adam-normal-bilstm-256'
    elif MODE=="simple" and BIDIRECTIONAL==1:
        checkpoint_dir = './training_checkpoints/adam-simple-bilstm-256'
    elif MODE=="SBT" and BIDIRECTIONAL==1:
        checkpoint_dir = './training_checkpoints/adam-SBT-256'
    
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    return encoder, decoder


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

    while(1):
        print(RED+"\n"+"="*80+RESET_COLOR)
        index = int(input("code number: "))
        if index == -1:
            break
        print(GREEN+"\nCode:"+RESET_COLOR)
        print(test_inputs[index])
        
        if BIDIRECTIONAL==0:
            print(GREEN+"Original comment:\n"+RESET_COLOR+test_outputs[index])
            predict = translate(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
            print(GREEN+"\nGreedy search prediction:\n"+RESET_COLOR+ predict)
            predict = beam_search(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, 3)
            print(GREEN+"\nBeam search prediction (k=3):\n"+RESET_COLOR+ predict)
        elif BIDIRECTIONAL==1:
            print(GREEN+"Original comment:\n"+RESET_COLOR+test_outputs[index])
            predict = translate_bilstm(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
            print(GREEN+"\nGreedy search prediction:\n"+RESET_COLOR+ predict)
            predict = beam_search_bilstm(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, 3)
            print(GREEN+"\nBeam search prediction (k=3):\n"+RESET_COLOR+ predict)
            
