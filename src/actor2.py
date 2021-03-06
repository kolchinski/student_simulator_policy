import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import logging
import os
import random
from IPython import embed

logging.basicConfig(level=logging.INFO)
train_dir = "./train_actor"
log_dir = "./actor_logs"
MAX_LENGTH = 100
HIDDEN_SIZE = 200
DROPOUT = 0.1
LR = 0.01


class Actor(object):
    def __init__(self, num_topics, hidden_size = 200, max_length = 100):

        self.num_topics = num_topics
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.build_pipeline()

    def build_pipeline(self):
        self.add_placeholders()

        with tf.variable_scope("actor"):
            self.pred_actions = self.build_network()

        self.objective = self.add_objective()
        self.train_op = tf.train.AdamOptimizer(self.lr_placeholder).minimize(-1.0 * self.objective)

    def add_placeholders(self):
        self.topics_placeholder = tf.placeholder(tf.int32,(None, MAX_LENGTH))
        self.answers_placeholder = tf.placeholder(tf.int32, (None, MAX_LENGTH))
        self.rewards_placeholder = tf.placeholder(tf.float32, (None, MAX_LENGTH))
        self.seq_lens_placeholder = tf.placeholder(tf.int32, (None))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder_with_default(DROPOUT, ())
        self.lr_placeholder = tf.placeholder_with_default(LR, ())

    def build_network(self):
        d = 1.0 - self.dropout_placeholder
        h = self.hidden_size

        cell = tf.contrib.rnn.LSTMCell(h)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = d)

        self.seqs = tf.one_hot(self.topics_placeholder * 2 + self.answers_placeholder, 2*self.num_topics)
        outputs, hidden_states = tf.nn.dynamic_rnn(
            cell=cell, inputs=self.seqs,
            sequence_length=self.seq_lens_placeholder, dtype=tf.float32,
            swap_memory=True)


        xav_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable("W", (self.hidden_size, self.num_topics), tf.float32, xav_init)
        b = tf.get_variable("b", (self.num_topics,), tf.float32, tf.constant_initializer(0.0))
        outputs_flat = tf.reshape(outputs, [-1, self.hidden_size])
        inner = tf.matmul(outputs_flat, w) + b

        # p(action) predicted for each time step, for each topic class - sum to 1
        self.probs = tf.nn.softmax(tf.reshape(inner, [-1, self.max_length, self.num_topics]))

        self.next_actions = tf.argmax(self.probs, axis=2)

        return self.next_actions

    def add_objective(self):

        topic_indicators = tf.one_hot(self.topics_placeholder, self.num_topics)
        topical_probs = tf.reduce_sum(self.probs * topic_indicators, 2)
        log_probs = tf.log(topical_probs)
        objectives = self.rewards_placeholder * log_probs
        masked_objectives = tf.boolean_mask(objectives, self.mask_placeholder)
        objective = tf.reduce_sum(masked_objectives)
        return objective

    def train_on_batch(self, session, rewards_batch, lens_batch, masks_batch, answers_batch, topics_batch):
        feed_dict = {self.rewards_placeholder: rewards_batch,
                     self.seq_lens_placeholder: lens_batch,
                     self.mask_placeholder: masks_batch,
                     self.answers_placeholder: answers_batch,
                     self.topics_placeholder: topics_batch}
        _, obj = session.run([self.train_op, self.objective], feed_dict=feed_dict)
        return obj

    def test_on_batch(self, session, rewards_batch, lens_batch, masks_batch, answers_batch, topics_batch):
        feed_dict = {self.rewards_placeholder: rewards_batch,
                     self.seq_lens_placeholder: lens_batch,
                     self.mask_placeholder: masks_batch,
                     self.answers_placeholder: answers_batch,
                     self.topics_placeholder: topics_batch}
        next_actions = session.run([self.next_actions], feed_dict=feed_dict)
        return next_actions


# Note: testing the actor with pregenerated sequences like this is not strictly
# correct, as the REINFORCE algorithm trains a policy with respect to the actions
# the policy *actually selects*, whereas here the policy is trained with respect
# to randomly taken actions
# Still, the fact that the actor learns something sensible based on this is a
# good sign for it working
def fake_sequences(num_seqs, num_topics=2, seq_len=50):
    learning_rates = np.ones(num_topics) * 0.2
    #learning_rates[1] = 1

    topics = []
    answers = []
    masks = []
    seq_lens = []
    rewards = []
    for i in range(num_seqs):
        seq_lens.append(seq_len)
        masks.append([1] * seq_len + [0] * (100 - seq_len))
        cur_topics = []
        cur_answers = []
        cur_rewards = []
        probs = np.zeros(num_topics)
        last_probs = np.copy(probs)
        for t in range(seq_len):
            topic = np.random.randint(num_topics)
            cur_topics.append(topic)
            answer = int(np.random.random() < probs[topic])
            cur_answers.append(answer)

            last_probs = np.copy(probs)
            if topic == 0 or probs[0] >= 0.8 and topic == 1 or probs[1] >= 0.8 and topic == 2:
                probs[topic] = min(1.0, probs[topic] + learning_rates[topic])
            #probs[topic] = min(1.0, probs[topic] + learning_rates[topic])
            #probs[topic] = probs[topic] + learning_rates[topic]
            cur_rewards.append(np.sum(probs) - np.sum(last_probs))
        topics.append(cur_topics + [0]*(100-seq_len))
        answers.append(cur_answers + [0]*(100-seq_len))
        rewards.append(cur_rewards + [0]*(100-seq_len))
    return topics, answers, masks, seq_lens, rewards


def main(_):
    print "Testing actor"
    topics, answers, masks, seq_lens, rewards = fake_sequences(20000, 3)
    actor = Actor(3, HIDDEN_SIZE, MAX_LENGTH)
    #embed()


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        #print topics
        for i in range(200):
            s,e = 100*i, 100*(i+1)
            obj = actor.train_on_batch(session, rewards[s:e], seq_lens[s:e], masks[s:e], answers[s:e], topics[s:e])
            print obj
        actions = actor.test_on_batch(session, rewards[s:e], seq_lens[s:e], masks[s:e], answers[s:e], topics[s:e])
        actionsArray = np.array(actions[0])
        zerosByTime = np.sum(actionsArray == 0, axis=0)
        onesByTime = np.sum(actionsArray == 1, axis=0)
        twosByTime =  np.sum(actionsArray == 2, axis=0)
        avgZeroPos = np.sum(np.arange(50) * zerosByTime[:50]) / np.sum(zerosByTime[:50])
        avgOnePos = np.sum(np.arange(50) * onesByTime[:50]) / np.sum(onesByTime[:50])
        avgTwoPos = np.sum(np.arange(50) * twosByTime[:50]) / np.sum(twosByTime[:50])
        print avgZeroPos, avgOnePos, avgTwoPos
        embed()
        #print actions


if __name__ == "__main__":
    tf.app.run()

