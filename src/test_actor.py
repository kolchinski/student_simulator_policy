# File for testing the actor class
# doesn't use real data.
import tensorflow as tf
import numpy as np
import actor
import random


# two different lessons:
# one with learning rate 0.2 per round
# one with learning rate 0.1 per round
batch_size = 5
seq_len = 8


# debug action variables

# Now check out what the model infers:


def run_model(model, session, test=False):
    critic_scores = np.zeros((batch_size, 2))  # how much we actually learn
    q_hist = np.zeros((batch_size, seq_len))
    correct_hist = np.zeros((batch_size, seq_len), dtype=np.bool)
    seq_lens = np.ones((batch_size,), dtype=np.int32)
    if test: model.action_probs = []
    for j in range(seq_len):
        extra_args = dict(collect_action_probs=True) if test else {}

        actions = model.get_next_action(session, q_hist, correct_hist, seq_lens, epsilon=0, **extra_args)
        learning = np.zeros((batch_size,))
        for an, action in enumerate(actions):
            if critic_scores[an, action] < 1:
                if action == 0:
                    learning[an] = 0.1
                else:
                    # learning[an] = 0.2
                    learning[an] = 0.25 - 0.02 * j  # more useful to learn early
                critic_scores[an, action] += learning[an]
            else:
                assert learning[an] == 0
            correct_hist[an, j] = (random.random() < critic_scores[an, action])

        q_hist[:, j] = actions
        if not test: model.apply_grad(session, learning)
        seq_lens += 1

    if test:
        print("Expected [ 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,]")
        print(q_hist)

        print("Avg Probs of action 1:")
        print("predicted that it should be close to 1 for first 5 time periods and quickly drop off.")
        array_probs = np.asarray(model.action_probs)
        print(array_probs[:, :, 1].mean(axis=1))  # in this case the batch is on axis 1....


        # print("All Probabilities:")
        # print(model.action_probs)


def test_system():
    model = actor.Actor(categories=2, cat_vec_len=seq_len, seq_len=seq_len)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(600):
            if i % 20 == 0:
                run_model(model, session, test=True)  # see behavior over time.
            run_model(model, session)

        run_model(model, session, test=True)

if __name__ == '__main__':
    test_system()

