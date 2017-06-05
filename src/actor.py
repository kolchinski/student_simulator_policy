import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
train_dir = "./train_actor"
log_dir = "./actor_logs"

def main():
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(log_dir +  "/log.txt")
    logging.getLogger().addHandler(file_handler)

    actor_model = Actor(100, 20)

    """
    with tf.Session() as sess:
        initialize_model(sess, qa)

        qa.train(sess, critic , train_dir)

        qa.evaluate_answer(sess, critic, log=True)
    """

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
    def __init__(self, categories, cat_vec_len=20, hidden_sz=50, seq_len=30, dropout=0.1):
        """
        
        :param categories: Number of learning categories
        :param cat_vec_len: embedding length for the category
        :param hidden_sz: size of lstm hidden state (x4)
        :param seq_len: max number of questions to ask
        """
        self.moving_avg = 0
        self.avg_counts = 0
        self.action_applied = False
        self.build_network(categories, cat_vec_len, hidden_sz, seq_len, dropout)

    def build_network(self, categories, cat_vec_len, hidden_sz, seq_len, dropout):
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
            hidden_1 = layers.fully_connected(lstm_index_i, num_outputs=categories, scope="FC1")
            self.final_hid = layers.fully_connected(hidden_1, num_outputs=categories,
                                                    activation_fn=None, scope="FC2")
            self.res = tf.nn.softmax(self.final_hid)


        self.action_gradient = tf.placeholder(tf.float32, [None, categories])
        # the minimize function for the adam loss has a "grad_loss" param taht is useful
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.res, self.action_gradient)

    def get_next_action(self, session, question_hist, correct_hist, cur_q_num):
        """
        :param question_hist: Questions given in the past (ndarray [Batch_sz])
        :param correct_hist: Whether the question was answered correctly ([Batch_sz] NDarray)
        :param step_nos: Step number to calculate each next step (ndarray [Batch_sz])
        :return: 
        """
        self.action_applied = True

        self.action_feed_dict = {
            self.q: question_hist,
            self.answ_correct: correct_hist,
            self.cur_q_number: cur_q_num,
        }
        action_probs, = session.run([self.res], feed_dict=self.action_feed_dict)
        next_action = np.argmax(action_probs, axis=1)
        return next_action

    def apply_grad(self, session,  action_perf):

        # first retrieve stored action
        if not self.action_applied:
            raise Exception("You must call Actor.get_next_action before applying another training step")
        self.action_applied = False

        action_perf = self.action_feed_dict.update({
            self.action_gradient: action_perf
        })

        session.run([self.train_op], feed_dict= action_perf)





