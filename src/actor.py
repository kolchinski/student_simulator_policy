import tensorflow as tf
import logging
import os

logging.basicConfig(level=logging.INFO)
train_dir = "./train_actor"
log_dir = "./actor_logs"

def main():
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_dir +  "/log.txt")
    logging.getLogger().addHandler(file_handler)

    with tf.Session() as sess:
        initialize_model(sess, qa)

        qa.train(sess, critic , train_dir)

        qa.evaluate_answer(sess, critic, log=True)

def initialize_model(session, model, train_dir=train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model



class Actor(object):
    def __init__(self, categories, cat_vec_len, hidden_sz=50, seq_len=30, dropout=0.1):
        """
        
        :param categories: Number of learning categories
        :param cat_vec_len: embedding length for the category
        :param hidden_sz: size of lstm hidden state (x4)
        :param seq_len: max number of questions to ask
        """

        xav_init = tf.contrib.layers.xavier_initializer()
        self.cat_indicies = tf
        # Embeddings
        self.q = tf.placeholder(tf.int32, (None, seq_len))
        self.cur_q_number = tf.placeholder(tf.int32, (None))  # the question for which we want an inference
        self.answ_correct = tf.placeholder(tf.float32, (None, seq_len))
        with tf.VariableScope("Actor"):
            embed = tf.get_variable("embed", (categories, cat_vec_len), xav_init)
            q_embed = tf.nn.embedding_lookup(embed, self.q)

            lstm_in = tf.concat([q_embed, self.answ_correct], axis=1)
            cell = tf.nn.rnn_cell.LSTMCell(hidden_sz)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0 - dropout)

            with tf.VariableScope("LSTM1"):
                lstm1_out, lstm1_states = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_in, sequence_length=seq_len,
                                                            dtype=tf.float32, swap_memory=True)

            lstm_index_i = lstm1_out[:, self.cur_q_number]
            with tf.VariableScope("FirstLinearLayer")






