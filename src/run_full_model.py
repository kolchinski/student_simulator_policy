
import tensorflow as tf
import numpy as np
import actor
import random


import logging

batch_size = 16
seq_len = 30
PRINT_EVERY = 10

def run_model(model, session, critic_fn, test=False):
    """
    runs an iteration of the actor model
    :return avg_learning: the average amount the actor learns in an episode
    """
    q_hist = np.zeros((batch_size, seq_len))
    correct_hist = np.zeros((batch_size, seq_len), dtype=np.bool)
    seq_lens = np.ones((batch_size,), dtype=np.int32)
    total_learning = np.zeros((batch_size, seq_len))  # all of the individual learning amounts for an episode

    if test: model.action_probs = []
    for j in range(seq_len):
        extra_args = dict(collect_action_probs=True) if test else {}

        actions = model.get_next_action(session, q_hist, correct_hist, seq_lens, epsilon=0, **extra_args)
        # learning = np.zeros((batch_size,))
        cur_learning, action_correct = critic_fn(q_hist, correct_hist, seq_lens, actions)
        correct_hist[:, j] = action_correct
        if j == 0:
            total_learning[:, 0] = cur_learning
        else:
            total_learning[:, j] = cur_learning - total_learning[:, j-1]

        q_hist[:, j] = actions
        if not test: model.apply_grad(session, cur_learning)
        seq_lens += 1

    return np.mean(np.sum(total_learning, axis=1))

def main(actor_episodes=10000):
    model = actor.Actor(categories=2, cat_vec_len=seq_len, seq_len=seq_len)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # train the critic


        # train the actor
        for i in range(actor_episodes):
            learning = run_model(model, session)
            if i % (actor_episodes // 10 ) == 0:
                run_model(model, session, test=True)
            if i % PRINT_EVERY == 0:


        run_model(model, session, test=True)



if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s    %(message)s', datefmt='%I:%M:%S', level=logging.INFO)
    file_handler = logging.FileHandler("model_perf.log")
    file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s    %(message)s', datefmt='%I:%M:%S'))
    logging.getLogger().addHandler(file_handler)
    main()
