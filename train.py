import os
from util import *
from config import *
from model import *
from tqdm import tqdm

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True) 

if __name__ == '__main__':
    code_train, comment_train, code_voc, comment_voc = read_pkl()    
    vocab_inp_size = len(code_voc)
    vocab_tar_size = len(comment_voc)
    max_length_inp = max(len(t) for t in code_train)
    max_length_targ = max(len(t) for t in comment_train)

    BUFFER_SIZE = len(code_train)
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    
    encoder, decoder= create_encoder_decoder(vocab_inp_size, vocab_tar_size, max_length_inp)

    optimizer = tf.optimizers.Adam(learning_rate=1e-3)  #tensorflow 2.0

    checkpoint_dir = getCheckpointDir()
    
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    _ = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    lossArray = np.array([])
    testAccuracy = []

    test_inputs, test_outputs = read_testset('./simplified_dataset/simplified_test.json')
    print('start training...')

    patientEarlyStopCount = 0
    EPOCHS = 100
    for epoch in range(1,EPOCHS+1):
        start = time.time()
        if ARCH == "lstm_lstm" or ARCH == "cnnlstm_lstm":
            hidden_h, hidden_c = encoder.initialize_hidden_state()
            hidden = [hidden_h, hidden_c]
        elif ARCH == "cnnbilstm_lstm":
            forward_h, forward_c, backward_h, backward_c = encoder.initialize_hidden_state()
            hidden = [forward_h, forward_c, backward_h, backward_c]
        elif  ARCH == "CODE-NN":
            pass

        total_loss = 0 
        code_train_batch = getBatch(code_train, BATCH_SIZE)
        comment_train_batch = getBatch(comment_train, BATCH_SIZE)
        dataset = [(code_train_batch[i], comment_train_batch[i]) for i in range(0, len(code_train_batch))]
        np.random.shuffle(dataset)
        
        for (batch, (inp, targ)) in enumerate(tqdm(dataset)):
            loss = 0
            with tf.GradientTape() as tape:
                if ARCH == "lstm_lstm" or ARCH == "cnnlstm_lstm" or ARCH == "cnnbilstm_lstm":
                    enc_output, enc_hidden_h, enc_hidden_c = encoder(inp, hidden)
                elif ARCH == "CODE-NN":
                    enc_output, enc_hidden_h, enc_hidden_c = tf.zeros((BATCH_SIZE, UNITS)), tf.zeros((BATCH_SIZE, UNITS)), tf.zeros((BATCH_SIZE, UNITS))

                dec_hidden = [enc_hidden_h, enc_hidden_c]

                dec_input = tf.expand_dims([comment_voc.index('<START>')] * BATCH_SIZE, 1)       

                # Teacher forcing - feeding the target as the next input
                for t in range(0, targ.shape[1]):
                    predictions, dec_hidden = decode_iterate(decoder, dec_input, dec_hidden, enc_output, inp)
                    loss += loss_function(targ[:, t], predictions)
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
                
            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss
            if ARCH == "CODE-NN":
                variables = decoder.variables
            else:
                variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))    
        lossArray = np.append(lossArray, (total_loss / N_BATCH) )    
        
        training_time = time.time() - start
        
        # calculate test accuracy
        total_bleu = 0
        for index, test in enumerate(test_inputs):
            predict = translate(test_inputs[index], encoder, decoder, code_voc, comment_voc, max_length_inp, max_length_targ)
            bleu_score = bleu(test_outputs[index], predict, 1)
            total_bleu += bleu_score
        total_bleu = total_bleu / len(test_inputs)
        testAccuracy.append(total_bleu)

        if epoch != 1 and total_bleu < testAccuracy[-2]:
            patientEarlyStopCount += 1
        if patientEarlyStopCount == 3:
            break

        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        output_f = open(checkpoint_dir+"/training_log", "a")
        if epoch == 1:
            print('Training time taken for 1 epoch {} sec\n'.format(training_time))
            output_f.write('Training time taken for 1 epoch {} sec\n'.format(training_time))
        
        if epoch == 1:
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

        print('Epoch {} Loss {:.4f}  Testing accuracy {:.4f}'.format(epoch, total_loss / N_BATCH, total_bleu))
        output_f.write('Epoch {} Loss {:.4f}  Testing accuracy {:.4f}\n'.format(epoch, total_loss / N_BATCH, total_bleu))
        output_f.close()
        
        epoch += 1
        
        
    # ======= recording the hyper-parameters of the models ===========
    f_parameter = open(checkpoint_dir+"/parameters", "a")
    f_parameter.write("EPOCHS="+str(epoch)+"\n")
    f_parameter.write("BATCH_SIZE="+str(BATCH_SIZE)+"\n")
    f_parameter.write("MODE="+MODE+"\n")
    f_parameter.write("ARCH="+ARCH+"\n")
    f_parameter.close()
