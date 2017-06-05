import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import logging
import os
import random

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
        self.num_cats = categories
        self._moving_avg = 0
        self._avg_counts = 0
        self._action_applied = False
        self.build_network(categories, cat_vec_len, hidden_sz, seq_len, dropout)

    def build_network(self, categories, cat_vec_len, hidden_sz, max_seq_len, dropout):
        xav_init = tf.contrib.layers.xavier_initializer()
        self.cat_indicies = tf
        # Embeddings
        self.q = tf.placeholder(tf.int32, (None, max_seq_len))
        self.answ_correct = tf.placeholder(tf.float32, (None, max_seq_len))
        self.seq_len = tf.placeholder(tf.int32, (None))
        with tf.variable_scope("Actor"):
            embed = tf.get_variable("embed", (categories, cat_vec_len), dtype=tf.float32
                                    , initializer=xav_init)
            q_embed = tf.nn.embedding_lookup(embed, self.q)

            answ_correct_3d = tf.reshape(self.answ_correct, (-1, max_seq_len, 1))
            lstm_in = tf.concat([q_embed, answ_correct_3d], axis=2)
            cell = tf.contrib.rnn.LSTMCell(hidden_sz)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)

            with tf.variable_scope("LSTM1"):
                lstm1_out, lstm1_state = tf.nn.dynamic_rnn(cell=cell, inputs=lstm_in, sequence_length=self.seq_len,
                                                            dtype=tf.float32, swap_memory=True)

            hidden_1 = layers.fully_connected(lstm1_state.h, num_outputs=categories, scope="FC1")
            self.final_hid = layers.fully_connected(hidden_1, num_outputs=categories,
                                                    activation_fn=None, scope="FC2")
            self.res = tf.nn.softmax(self.final_hid)


        # now build the loss and gradient backprop
        self.action_mask = tf.placeholder(tf.bool, [None, categories])
        self.action_gradient = tf.placeholder(tf.float32, [None, categories])
        # the minimize function for the adam loss has a "grad_loss" param that is useful
        opt = tf.train.AdamOptimizer(1e-2)
        mask_res = self.res * tf.cast(self.action_mask, dtype=tf.float32)
        self.train_op = opt.minimize(mask_res, grad_loss=-self.action_gradient)

    def get_next_action(self, session, question_hist, correct_hist, seq_lens,
                        collect_action_probs=False):
        """
        :param question_hist: Questions given in the past (ndarray [Batch_sz, N])
        :param correct_hist: Whether the question was answered correctly ([Batch_sz, N] NDarray)
        :return: 
        """
        self._action_applied = True

        self.action_feed_dict = {
            self.q: question_hist,
            self.answ_correct: correct_hist,
            self.seq_len: np.copy(seq_lens),
        }

        """
        if random.random() < epsilon:
            self.next_action = np.random.choice(np.arange(self.num_cats), seq_lens.shape)
            return self.next_action
        """

        action_probs, = session.run([self.res], feed_dict=self.action_feed_dict)

        if collect_action_probs:
            self.action_probs.append(action_probs)

        # self.next_action = np.argmax(action_probs, axis=1)
        next_a = []
        for single_a_probs in action_probs:
            next_a.append(np.random.choice(self.num_cats, 1, p=single_a_probs)[0])
        self.next_action = np.asarray(next_a)
        return self.next_action

    def apply_grad(self, session, action_perf):

        # first retrieve stored action
        if not self._action_applied:
            raise Exception("You must call Actor.get_next_action before applying another training step")
        self._action_applied = False

        # get reference avg performance
        self._moving_avg = (self._moving_avg * self._avg_counts + np.sum(action_perf)) / \
                           (self._avg_counts + len(action_perf))
        self._avg_counts += len(action_perf)

        # now normalize to this average perf
        action_vec = np.zeros((len(action_perf), self.num_cats), dtype=np.float32)
        action_mask = np.zeros_like(action_vec, dtype=np.bool)
        action_delta = action_perf - self._moving_avg

        for i, val in enumerate(action_delta):
            action_vec[i, self.next_action[i]] = val
            action_mask[i, self.next_action[i]] = True

        self.action_feed_dict.update({
            self.action_gradient: action_vec,
            self.action_mask: action_mask
        })

        _ = session.run([self.train_op], feed_dict=self.action_feed_dict)






