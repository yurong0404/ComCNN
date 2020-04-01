import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
from util import *
from param import *
from model import *
from tqdm import tqdm

if __name__ == '__main__':
    code_train, comment_train, code_voc, comment_voc = read_pkl()
    vocab_inp_size = len(code_voc)
    vocab_tar_size = len(comment_voc)
    max_length_inp = max(len(t) for t in code_train)
    max_length_targ = max(len(t) for t in comment_train)

    BUFFER_SIZE = len(code_train)
    N_BATCH = BUFFER_SIZE//BATCH_SIZE

    encoder = BidirectionalEncoder(vocab_inp_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)
    decoder = BidirectionalDecoder(vocab_tar_size, EMBEDDING_DIM, UNITS, BATCH_SIZE)

    optimizer = tf.optimizers.Adam(learning_rate=1e-3)  #tensorflow 2.0

    checkpoint_dir = getCheckpointDir()

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    lossArray = np.array([])
    testAccuracy = []

    test_inputs, test_outputs = read_testset()
    print('start training...')

    EPOCHS = 50
    for epoch in range(1,EPOCHS+1):
        start = time.time()
        hidden = encoder.initialize_hidden_state()

        total_loss = 0 
        code_train_batch = getBatch(code_train, BATCH_SIZE)
        comment_train_batch = getBatch(comment_train, BATCH_SIZE)
        dataset = [(code_train_batch[i], comment_train_batch[i]) for i in range(0, len(code_train_batch))]
        np.random.shuffle(dataset)
        
        for (batch, (inp, targ)) in enumerate(tqdm(dataset)):
            loss = 0
            with tf.GradientTape() as tape:
                enc_output, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = encoder(inp, hidden, hidden, hidden, hidden)
                dec_hidden = [enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c]
                dec_input = tf.expand_dims([comment_voc.index('<START>')] * BATCH_SIZE, 1)       

                # Teacher forcing - feeding the target as the next input
                for t in range(0, targ.shape[1]):
                    predictions, dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c = decoder(dec_input, dec_hidden, enc_output)
                    dec_hidden = [dec_forward_h, dec_forward_c, dec_backward_h, dec_backward_c]
                    loss += loss_function(targ[:, t], predictions)
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            
        lossArray = np.append(lossArray, (total_loss / N_BATCH) )    
        
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        # calculate test accuracy
        total_bleu = 0
        for index, test in enumerate(test_inputs):
            predict = translate_bilstm(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
            bleu_score = bleu(test_outputs[index], predict, 1)
            total_bleu += bleu_score
        total_bleu = total_bleu / len(test_inputs)
        testAccuracy.append(total_bleu)
        
        output_f = open(checkpoint_dir+"/training_log", "a")
        if epoch == 1:
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            output_f.write('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        print('Epoch {} Loss {:.4f}  Testing accuracy {:.4f}'.format(epoch, total_loss / N_BATCH, total_bleu))
        output_f.write('Epoch {} Loss {:.4f}  Testing accuracy {:.4f}\n'.format(epoch, total_loss / N_BATCH, total_bleu))
        output_f.close()
        
        epoch += 1
        
        
    # ======= recording the hyper-parameters of the models ===========
    f_parameter = open(checkpoint_dir+"/parameters", "a")
    f_parameter.write("EPOCHS="+str(epoch)+"\n")
    f_parameter.write("BATCH_SIZE="+str(BATCH_SIZE)+"\n")
    f_parameter.write("MODE="+MODE+"\n")
    f_parameter.close()
