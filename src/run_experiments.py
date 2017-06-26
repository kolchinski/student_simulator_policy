import numpy as np
import tensorflow as tf
from actor2 import Actor
import dkt_tf
from IPython import embed

NUM_BATCHES = 10
BATCH_SIZE = 5

def train_actor_on_model(actor_session, model_session, actor, model):
    max_length = model.max_length
    print 'Training actor; model max length is', max_length


    for batch in range(NUM_BATCHES):
        print("Training on batch", batch)

        #TODO: either debatchify this or make the actor probabilistic instead of deterministic
        #TODO: or actually this might be fine because the simulated correct/incorrect is randomized
        rewards_batch = np.zeros((BATCH_SIZE, max_length))
        lens_batch = np.zeros(BATCH_SIZE)
        masks_batch = np.zeros((BATCH_SIZE, max_length))
        answers_batch = np.zeros((BATCH_SIZE, max_length))
        topics_batch = np.zeros((BATCH_SIZE, max_length))

        # NumPy array of shape BATCH_SIZE x num_topics
        last_probs = model.next_probs(model_session, lens_batch, masks_batch, answers_batch, topics_batch)
        print "Original probabilities", last_probs

        for index in range(max_length):

            # Get actor's decision for next time step's actions
            actions = actor.test_on_batch(
                actor_session, rewards_batch, lens_batch, masks_batch, answers_batch, topics_batch)
            actions_array = np.array(actions[0])
            next_actions = actions_array[:,index]
            topics_batch[:, index] = next_actions
            masks_batch[:, index] = 1
            lens_batch += 1

            #print 'Next actions:', actions

            # Get model's predictions for consequences of those actions
            next_probs, topical_probs = model.next_probs_with_topical(
                model_session,lens_batch, masks_batch, answers_batch, topics_batch)
            next_topical_probs = topical_probs[:, index]
            #print 'Next probabilities of correct answers:', next_probs

            skill_gains = np.sum(next_probs - last_probs, axis=1)
            print(index, next_topical_probs)
            rewards_batch[:, index] = skill_gains
            train_masks = np.zeros((BATCH_SIZE, max_length))
            train_masks[:, index] = 1

            #Simulate answers based on probabilities
            corrects = np.random.rand(BATCH_SIZE) < next_topical_probs
            answers_batch[:, index] = corrects

            #TODO: Might be able to combine training and testing actor (for next time step) and save a bunch of ops
            actor.train_on_batch(actor_session, rewards_batch, lens_batch, train_masks, answers_batch, topics_batch)

#TODO: Refactor so that test and train actor on model don't reuse so much code
def test_actor_on_model(actor_session, model_session, actor, model):
    max_length = model.max_length
    print 'Testing actor; model max length is', max_length

    all_skill_gains = 0

    for batch in range(NUM_BATCHES):
        print("Testing on batch", batch)

        rewards_batch = np.zeros((BATCH_SIZE, max_length))
        lens_batch = np.zeros(BATCH_SIZE)
        masks_batch = np.zeros((BATCH_SIZE, max_length))
        answers_batch = np.zeros((BATCH_SIZE, max_length))
        topics_batch = np.zeros((BATCH_SIZE, max_length))

        # NumPy array of shape BATCH_SIZE x num_topics
        last_probs = model.next_probs(model_session, lens_batch, masks_batch, answers_batch, topics_batch)
        original_probs = last_probs

        print "Original probabilities", original_probs

        for index in range(max_length):

            # Get actor's decision for next time step's actions
            actions = actor.test_on_batch(
                actor_session, rewards_batch, lens_batch, masks_batch, answers_batch, topics_batch)
            actions_array = np.array(actions[0])
            next_actions = actions_array[:,index]
            topics_batch[:, index] = next_actions
            masks_batch[:, index] = 1
            lens_batch += 1

            # Get model's predictions for consequences of those actions
            next_probs, topical_probs = model.next_probs_with_topical(
                model_session,lens_batch, masks_batch, answers_batch, topics_batch)
            next_topical_probs = topical_probs[:, index]

            skill_gains = np.sum(next_probs - last_probs, axis=1)
            rewards_batch[:, index] = skill_gains

            #Simulate answers based on probabilities
            corrects = np.random.rand(BATCH_SIZE) < next_topical_probs
            answers_batch[:, index] = corrects

        total_skill_gains = np.sum(next_probs - original_probs, axis=1)
        print "Total skill gains:", total_skill_gains
        all_skill_gains += np.sum(total_skill_gains)

    avg_skill_gain = all_skill_gains / (NUM_BATCHES * BATCH_SIZE)
    return avg_skill_gain


class RandomActor:
    def __init__(self, num_topics):
        self.num_topics = num_topics

    def test_on_batch(self, session, rewards, lens, masks, answers, topics):
        num_batches = len(topics)
        seq_len = len(topics[0])
        return [np.random.randint(0, self.num_topics, size=(num_batches,seq_len))]



def main(_):
    print "Running experiments"

    model_session = tf.Session()
    model_session.run(tf.global_variables_initializer())
    #We need to explicitly initialize local variables to use
    #TensorFlow's AUC function for some reason...
    model_session.run(tf.local_variables_initializer())
    model1, model2 = dkt_tf.get_paired_models(model_session)
    print model1, model2

    actor_session = tf.Session()
    actor = Actor(124)
    actor_session.run(tf.global_variables_initializer())
    train_actor_on_model(actor_session, model_session, actor, model1)

    random_actor = RandomActor(124)

    avg_skill_gain = test_actor_on_model(actor_session, model_session, actor, model2)
    print "Average skill gain for real actor", avg_skill_gain


    avg_random_skill_gain = test_actor_on_model(actor_session, model_session, random_actor, model2)
    print "Average skill gain for random actor", avg_random_skill_gain

    model_session.close()
    actor_session.close()


if __name__ == "__main__":
    tf.app.run()
