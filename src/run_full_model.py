import tensorflow as tf
import numpy as np
import random
import logging

import actor
import dkt_tf

MAX_LENGTH = 100
LR = 0.001
HIDDEN_SIZE = 200
BATCH_SIZE = 128
MAX_EPOCHS = 10
DROPOUT = 0.3
TRAIN_SPLIT = 0.7

PRINT_EVERY = 2


model_learning_res = None

def run_model(model, session, critic_fn, test=False, collect_extra_data=False):
    """
    runs an iteration of the actor model
    :return avg_learning: the average amount that the actor learned in an episode
    """

    batch_size = BATCH_SIZE
    seq_len = MAX_LENGTH

    q_hist = np.zeros((batch_size, seq_len), dtype=np.int32)
    correct_hist = np.zeros((batch_size, seq_len), dtype=np.bool)
    seq_lens = np.zeros((batch_size,), dtype=np.int32)
    total_learning = np.zeros((batch_size, seq_len))  # all of the individual learning amounts for an episode
    delta_learning = np.zeros((batch_size, seq_len))

    # Alex's stats
    all_Qs_stats = np.zeros((batch_size, seq_len, num_topics))

    cur_learning, action_correct = critic_fn(session, seq_lens, correct_hist, q_hist)
    all_Qs_stats[:, 0] = cur_learning

    total_learning[:, 0] = cur_learning.sum(axis=1)
    seq_lens += 1

    if test: model.action_probs = []
    for j in range(1, seq_len):
        extra_args = dict(collect_action_probs=True) if test else {}

        # get actor actions
        actions = model.get_next_action(session, q_hist, correct_hist, seq_lens, epsilon=0, **extra_args)
        q_hist[:, j - 1] = actions

        # get critic scores
        cur_learning, action_correct = critic_fn(session, seq_lens + 1, correct_hist, q_hist)

        # process critic answers
        correct_hist[:, j - 1] = action_correct
        total_learning[:, j] = cur_learning.sum(axis=1)
        delta_learning[:, j] = total_learning[:, j] - total_learning[:, j-1]

        # Alex's stats
        all_Qs_stats[:, j] = cur_learning

        # apply actor gradient
        if not test: model.apply_grad(session, cur_learning.sum(axis=1))
        seq_lens += 1

    if collect_extra_data:
        return np.mean(np.sum(delta_learning, axis=1)), all_Qs_stats[:, -1] - all_Qs_stats[:, 0]

    return np.mean(np.sum(delta_learning, axis=1))


class CriticWrapper(object):
    def __init__(self, critic):
        self.critic = critic

    def get_next_info(self, session, lens, prev_answers, prev_topics):
        tiled_range = np.tile(np.arange(MAX_LENGTH).reshape((1, -1)), (BATCH_SIZE, 1))
        mask = (tiled_range < lens.reshape((-1, 1)))

        new_probs = self.critic.next_probs(session, lens, mask, prev_answers, prev_topics)
        B, cats = new_probs.shape

        B_rng = np.arange(B, dtype=np.int32)
        cur_topics = prev_topics[B_rng, lens]
        topic_probs = new_probs[B_rng, cur_topics]

        # print(new_probs.shape, topic_probs.shape)
        new_correctness = np.random.random(topic_probs.shape) < topic_probs
        return new_probs, new_correctness


def main(actor_episodes=40):
    logging.info("tf version " + tf.__version__)

    # initialize critic stuff
    global num_topics  # Hack to get Alex's data
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


        # all of the extra data to collect
        # dimmensions: policy num (0=random, 1=actor), critic (0=train), people, topics
        alexs_data = - np.ones((2, 2, 256, num_topics))

        # Collect data on random model
        for model_no, c_model in enumerate((train_critic, dev_critic)):
            for batch_num in range(256 // BATCH_SIZE):
                learning, res = run_model(model, session, c_model.get_next_info, test=True, collect_extra_data=True)
                logging.info("random policy score {} {}: {}".format(model_no, batch_num, learning))
                alexs_data[0, model_no, BATCH_SIZE * batch_num: (batch_num + 1) * BATCH_SIZE] = res

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

        for model_no, c_model in enumerate((train_critic, dev_critic)):
            for batch_num in range(256 // BATCH_SIZE):
                learning, res = run_model(model, session, c_model.get_next_info, test=True, collect_extra_data=True)
                logging.info("actor policy score {} {}: {}".format(model_no, batch_num, learning))
                alexs_data[1, model_no, BATCH_SIZE * batch_num: (batch_num + 1) * BATCH_SIZE] = res

        # now to store the data
        np.save("alex_stats.npy", alexs_data)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s    %(message)s', datefmt='%I:%M:%S', level=logging.INFO)
    file_handler = logging.FileHandler("model_perf.log")
    formatter = logging.Formatter(fmt='%(asctime)s    %(message)s', datefmt='%I:%M:%S')
    logging.getLogger().handlers[0].setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    try:
        main()
    except BaseException as e:
        import traceback
        logging.error(traceback.format_exc())
        raise e
