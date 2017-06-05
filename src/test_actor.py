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
critic_scores = np.zeros((batch_size, 2))

seq_len=10


if __name__ == '__main__':
    model = actor.Actor(categories=2, cat_vec_len=10, seq_len=seq_len)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for i in range(100):
            q_hist = np.zeros((batch_size, seq_len))
            correct_hist = np.zeros((batch_size, seq_len), dtype=np.bool)
            seq_lens = np.ones((batch_size,))

            for j in range(seq_len):
                actions = model.get_next_action(session, q_hist, correct_hist, seq_lens, epsilon=0.2)
                learning = np.zeros((batch_size,))
                for an, action in enumerate(actions):
                    if critic_scores[an, action] >= 1:
                        continue # already a score of 0
                    if action == 0:
                        learning[an] = 0.2
                    else:
                        learning[an] = 0.1
                    critic_scores[an, action] += learning[an]
                    correct_hist[j, an] = (random.random() < critic_scores[an, action])

                q_hist[:, j] = actions
                model.apply_grad(session, learning)
                seq_lens += 1

        # Now check out what the model infers:
        q_hist = np.zeros((batch_size, seq_len))
        correct_hist = np.zeros((batch_size, seq_len), dtype=np.bool)
        seq_lens = np.ones((batch_size,))
        for j in range(seq_len):
            actions = model.get_next_action(session, q_hist, correct_hist, seq_lens, epsilon=0)
            learning = np.zeros((batch_size,))
            for an, action in enumerate(actions):
                if critic_scores[an, action] >= 1:
                    continue # already a score of 0
                if action == 0:
                    learning[an] = 0.2
                else:
                    learning[an] = 0.1
                critic_scores[an, action] += learning[an]
                correct_hist[j, an] = (random.random() < critic_scores[an, action])

            q_hist[j] = actions
            model.apply_grad(session, learning)
            seq_lens += 1

        print("Expected [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]")
        print(q_hist)







