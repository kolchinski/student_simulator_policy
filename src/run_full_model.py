import tensorflow as tf
import numpy as np
import random
import logging

import actor
import dkt_tf

MAX_LENGTH = 100
LR = 0.001
HIDDEN_SIZE = 200
BATCH_SIZE = 32
MAX_EPOCHS = 10
DROPOUT = 0.3
TRAIN_SPLIT = 0.7

batch_size = BATCH_SIZE
seq_len = MAX_LENGTH
PRINT_EVERY = 10

def run_model(model, session, critic_fn, test=False):
    """
    runs an iteration of the actor model
    :return avg_learning: the average amount that the actor learned in an episode
    """
    q_hist = np.zeros((batch_size, seq_len), dtype=np.int32)
    correct_hist = np.zeros((batch_size, seq_len), dtype=np.bool)
    seq_lens = np.ones((batch_size,), dtype=np.int32)
    total_learning = np.zeros((batch_size, seq_len))  # all of the individual learning amounts for an episode

    if test: model.action_probs = []
    for j in range(seq_len):
        extra_args = dict(collect_action_probs=True) if test else {}

        # get actor actions
        actions = model.get_next_action(session, q_hist, correct_hist, seq_lens, epsilon=0, **extra_args)
        q_hist[:, j] = actions

        # get critic scores
        cur_learning, action_correct = critic_fn(session, seq_lens, correct_hist, q_hist)

        # process critic answers
        correct_hist[:, j] = action_correct
        if j == 0:
            total_learning[:, 0] = cur_learning
        else:
            total_learning[:, j] = cur_learning - total_learning[:, j-1]

        # apply actor gradient
        if not test: model.apply_grad(session, cur_learning)
        seq_lens += 1

    return np.mean(np.sum(total_learning, axis=1))


class CriticWrapper(object):
    def __init__(self, critic):
        self.critic = critic

    def get_next_info(self, session, lens, prev_answers, prev_topics):
        tiled_range = np.tile(np.arange(MAX_LENGTH).reshape((1, -1)), (BATCH_SIZE, 1))
        mask = (tiled_range < lens.reshape((-1, 1)))

        new_probs_raw = self.critic.next_probs(session, lens, mask, prev_answers, prev_topics)
        B, _, cats = new_probs_raw.shape
        new_probs = new_probs_raw.reshape((B, cats))

        B_rng = np.arange(B, dtype=np.int32)
        cur_topics = prev_topics[B_rng, seq_len - 1]
        topic_probs = new_probs[B_rng, cur_topics]

        # print(new_probs.shape, topic_probs.shape)
        new_correctness = np.random.random(topic_probs.shape) < topic_probs
        return new_probs.sum(axis=1), new_correctness


def main(actor_episodes=10000):
    logging.info("tf version " + tf.__version__)

    # initialize critic stuff
    topics, answers, num_topics = dkt_tf.read_assistments_data(dkt_tf.DATA_LOC)
    full_data = dkt_tf.load_data(topics, answers, num_topics)

    # initialize actor stuff
    model = actor.Actor(categories=num_topics, cat_vec_len=num_topics // 2, seq_len=MAX_LENGTH,
                        hidden_sz=100)

    with tf.Session() as session:
        # session.run(tf.global_variables_initializer())
        # session.run(tf.local_variables_initializer())  trained paired models does this internally

        # train the critic
        model1, model2 = dkt_tf.train_paired_models(session, full_data, num_topics)
        train_critic = CriticWrapper(model1)
        dev_critic = CriticWrapper(model2)

        # train the actor
        for t in range(actor_episodes):
            learning = run_model(model, session, train_critic.get_next_info)
            if t % PRINT_EVERY == 0:
                logging.info("    t={}: train policy score: {}".format(t, learning))
            if t % (actor_episodes // 10) == 0:
                dev_learning = run_model(model, session, dev_critic.get_next_info, test=True)
                logging.info("t={}: Eval policy score: {}".format(t, dev_learning))

        dev_learning = run_model(model, session, dev_critic.get_next_info, test=True)
        logging.info("Final eval network policy score {}".format(dev_learning))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s    %(message)s', datefmt='%I:%M:%S', level=logging.INFO)
    file_handler = logging.FileHandler("model_perf.log")
    file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s    %(message)s', datefmt='%I:%M:%S'))
    logging.getLogger().addHandler(file_handler)
    main()
