from util import *
from model import *
from param import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def read_model(encoder, decoder):
    if MODE=="normal":
        checkpoint_dir = './training_checkpoints/adam-normal-256'
    elif MODE=="simple":
        checkpoint_dir = './training_checkpoints/adam-simple-256'
    elif MODE=="SBT":
        checkpoint_dir = './training_checkpoints/adam-SBT-256'
    
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

    return encoder, decoder


def read_testset():
    f = open('./simplified_dataset/simplified_test.json')
    inputs = f.readlines()
    f.close()
    test_inputs = []
    test_outputs = []

    for pair in inputs:
        pair = json.loads(pair)
        test_inputs.append(pair['code'])
        test_outputs.append(pair['nl'])
    
    return test_inputs, test_outputs


if __name__ == '__main__':
    print("Reading "+MODE+" model...")
    code_voc, comment_voc, vocab_inp_size, vocab_tar_size, max_length_inp, max_length_targ = read_pkl()
    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    encoder, decoder = read_model(encoder, decoder)
    test_inputs, test_outputs = read_testset()

    while(1):
        print(RED+"\n"+"="*80+RESET_COLOR)
        index = int(input("code number: "))
        if index == -1:
            break
        print(GREEN+"\nCode:"+RESET_COLOR)
        print(test_inputs[index])
        print(GREEN+"Original comment:\n"+RESET_COLOR+test_outputs[index])
        predict = translate(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
        print(GREEN+"\nGreedy search prediction:\n"+RESET_COLOR+ predict)
        predict = beam_search(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ, 3)
        print(GREEN+"\nBeam search prediction (k=3):\n"+RESET_COLOR+ predict)