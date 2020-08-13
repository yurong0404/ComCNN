import os
import time
from util import *
from config import *
from model import *
from tqdm import tqdm

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def calculate_test_acc(test_data, encoder, decoder, train_data):
    total_bleu = 0
    for data in tqdm(test_data):
        predict = greedy_search(data['code'], encoder, decoder, train_data)
        bleu_score = bleu(data['comment'], predict, 1)
        total_bleu += bleu_score
    total_bleu = total_bleu / len(test_data)
    return total_bleu


if __name__ == '__main__':
    train_data = read_train_pkl()
    test_data = read_testset()
    data_size = len(train_data['code'])
    encoder, decoder = create_model(
        train_data['code_voc_size'],
        train_data['com_voc_size'],
        train_data['max_length_code']
    )
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    checkpoint_dir = get_checkpoint_dir()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=encoder,
        decoder=decoder
    )
    # try to restore the half-trained model, if fail, that's also ok
    _ = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    lossArray = np.array([])
    test_accuracy = []
    print('start training...')
    patient_early_stop_counter = 0
    MAX_EPOCHS = 100

    for epoch in range(1, MAX_EPOCHS+1):
        start_time = time.time()
        code_train_batch = get_batch(train_data['code'], BATCH_SIZE)
        comment_train_batch = get_batch(train_data['comment'], BATCH_SIZE)
        dataset = list(zip(code_train_batch, comment_train_batch))
        np.random.shuffle(dataset)
        hidden = encoder.initialize_hidden_state(batch_sz=BATCH_SIZE)
        total_loss = 0

        for (inp, targ) in tqdm(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                dec_hidden = []
                enc_output, enc_hidden_h, enc_hidden_c = encoder(inp, hidden)
                dec_hidden = [enc_hidden_h, enc_hidden_c]
                start_tag_index = train_data['comment_voc'].index('<START>')
                dec_input = tf.expand_dims([start_tag_index] * BATCH_SIZE, 1)

                for t in range(0, targ.shape[1]):
                    predictions, dec_hidden_h, dec_hidden_c = decoder(dec_input, dec_hidden, enc_output)
                    dec_hidden = [dec_hidden_h, dec_hidden_c]
                    loss += loss_function(targ[:, t], predictions)
                    dec_input = tf.expand_dims(targ[:, t], 1)

            total_loss += loss
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        training_time = time.time() - start_time
        test_bleu = calculate_test_acc(test_data, encoder, decoder, train_data)
        test_accuracy.append(test_bleu)

        if epoch != 1 and test_bleu < test_accuracy[-2]:
            patient_early_stop_counter += 1

        # early stop the training process
        if patient_early_stop_counter == 3:
            break

        checkpoint.save(file_prefix=checkpoint_prefix)

        output_f = open(checkpoint_dir+"/training_log", "a")
        if epoch == 1:
            print('Training time taken for 1 epoch {} sec\n'.format(training_time))
            output_f.write('Training time taken for 1 epoch {} sec\n'.format(training_time))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))

        print('Epoch {} Loss {:.4f}  Testing accuracy {:.4f}'.format(epoch, total_loss / data_size, test_bleu))
        output_f.write('Epoch {} Loss {:.4f}  Testing accuracy {:.4f}\n'.format(epoch, total_loss / data_size, test_bleu))
        output_f.close()

    # ======= recording the hyper-parameters of the models ===========
    log_file = open(checkpoint_dir+"/parameters", "a")
    log_file.write("EPOCHS="+str(epoch)+"\n")
    log_file.write("BATCH_SIZE="+str(BATCH_SIZE)+"\n")
    log_file.write("ARCH="+ARCH+"\n")
    log_file.close()
