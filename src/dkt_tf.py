from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import defaultdict
from IPython import embed
import matplotlib.pyplot as plt


DATA_LOC = './assistments.txt'
MAX_LENGTH = 100
LR = 0.001
HIDDEN_SIZE = 200
BATCH_SIZE = 32
MAX_EPOCHS = 10
DROPOUT = 0.3
TRAIN_SPLIT = 0.7

def read_assistments_data(loc):
    with open(loc) as f:
        data = [[int(y) for y in x.strip().split()] for x in f.readlines()]

    topic_seqs = defaultdict(list)
    answer_seqs = defaultdict(list)
    topics = set()

    for d in data:
        student_id = d[0]
        if(len(topic_seqs[student_id])) >= MAX_LENGTH: continue
        topic_seqs[student_id].append(d[1])
        answer_seqs[student_id].append(d[2])
        topics.add(d[1])

    #keys = topic_seqs.keys()
    #np.random.shuffle(keys)
    #return [topic_seqs[k] for k in keys], [answer_seqs[k] for k in keys], max(topics) + 1
    #topics may not be consecutively numbered but assume they are
    order = list(topic_seqs.keys())

    return [topic_seqs[x] for x in order], [answer_seqs[x] for x in order], max(topics) + 1


# Takes jagged arrays of topic and answer sequences
# Pads them into square arrays; returns that and a mask
def load_data(topic_seqs, answer_seqs, num_topics):
    assert(len(topic_seqs) == len(answer_seqs))
    N = len(topic_seqs)

    topics_padded = []
    answers_padded = []
    masks = []
    embeddings = np.zeros((N, MAX_LENGTH, num_topics*2))
    seq_lens = []

    for i in range(N):
        assert(len(topic_seqs[i]) == len(answer_seqs[i]))
        seq_len = len(topic_seqs[i])
        seq_lens.append(seq_len)
        padding = [0] * (MAX_LENGTH - seq_len)
        topics_padded.append(topic_seqs[i] + padding)
        answers_padded.append(answer_seqs[i] + padding)
        masks.append([1] * seq_len + padding)
        #TODO: we can actually modify the code so that we predict the
        #probabilities of correct answers *after* the sequence ends as well.
        #This requires modifying some shapes as well as doing one more step
        #in this loop
        for j in range(seq_len - 1):
            # Make the one-hot representation; for each sequence and time step,
            # 2*n_topics length vector, one-hot for topic/answer index
            one_hot_index = topic_seqs[i][j] * 2 + answer_seqs[i][j]
            embeddings[i][j + 1][one_hot_index] = 1

    #padded topics and answers not actually used, could remove?
    topics = topics_padded
    answers = answers_padded
    return embeddings, seq_lens, masks, answers, topics

class DKTModel(object):


    def add_placeholders(self):
        self.topics_placeholder = tf.placeholder(tf.int32,(None, MAX_LENGTH))
        self.answers_placeholder = tf.placeholder(tf.int32, (None, MAX_LENGTH))
        self.seqs_placeholder = tf.placeholder(tf.float32, (None, self.max_length, 2 * self.num_topics))
        self.seq_lens_placeholder = tf.placeholder(tf.int32, (None))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder_with_default(DROPOUT, ())
        self.lr_placeholder = tf.placeholder_with_default(LR, ())


    def data_pipeline(self):
        d = 1.0 - self.dropout_placeholder
        h = self.hidden_size

        cell = tf.contrib.rnn.LSTMCell(h)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = d)

        outputs, hidden_states = tf.nn.dynamic_rnn(
            cell=cell, inputs=self.seqs_placeholder,
            sequence_length=self.seq_lens_placeholder, dtype=tf.float32,
            swap_memory=True)


        xav_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable("W", (self.hidden_size, self.num_topics), tf.float32, xav_init)
        b = tf.get_variable("b", (self.num_topics,), tf.float32, tf.constant_initializer(0.0))
        outputs_flat = tf.reshape(outputs, [-1, self.hidden_size])
        inner = tf.matmul(outputs_flat, w) + b

        # p(correct) predicted for each time step, for each topic class
        self.probs = tf.sigmoid(tf.reshape(inner, [-1, self.max_length, self.num_topics]))

        topic_indicators = tf.one_hot(self.topics_placeholder, self.num_topics)
        self.topical_probs = tf.reduce_sum(self.probs * topic_indicators, 2)

        return self.topical_probs

    def evaluate_probs(self):
        #Probabilities of getting each class at each time correct, for reference
        self.guesses = tf.to_int32(tf.round(self.topical_probs))
        corrects = tf.to_int32(tf.equal(self.guesses, self.answers_placeholder))
        masked_corrects = tf.boolean_mask(corrects, self.mask_placeholder)
        num_correct = tf.reduce_sum(masked_corrects)

        self.auc_score, self.auc_op = tf.metrics.auc(self.answers_placeholder, self.topical_probs)

        # How many total examples in this batch - for the denominator of # correct
        self.num_total = tf.reduce_sum(tf.to_int32(self.mask_placeholder))

        return num_correct

    def find_loss(self, probs, labels):
        losses = probs - tf.to_float(labels)
        masked_losses = tf.boolean_mask(losses, self.mask_placeholder)

        #Calculate mean-squared error
        total_loss = tf.reduce_sum(masked_losses**2)
        return total_loss


    def setup_system(self):
        self.add_placeholders()

        with tf.variable_scope("dkt"):
            self.topical_probs = self.data_pipeline()
            self.num_correct = self.evaluate_probs()

        self.loss = self.find_loss(self.topical_probs, self.answers_placeholder)
        self.train_op = tf.train.AdamOptimizer(self.lr_placeholder).minimize(self.loss)

        self.saver = tf.train.Saver()


    def train_on_batch(self, session, seqs_batch, lens_batch, masks_batch, answers_batch, topics_batch):
        feed_dict = {self.seqs_placeholder: seqs_batch,
                     self.seq_lens_placeholder: lens_batch,
                     self.mask_placeholder: masks_batch,
                     self.answers_placeholder: answers_batch,
                     self.topics_placeholder: topics_batch}
        _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def test_on_batch(self, session, seqs_batch, lens_batch, masks_batch, answers_batch, topics_batch):
        feed_dict = {self.seqs_placeholder: seqs_batch,
                     self.seq_lens_placeholder: lens_batch,
                     self.mask_placeholder: masks_batch,
                     self.answers_placeholder: answers_batch,
                     self.topics_placeholder: topics_batch}
        num_correct, num_total, auc = \
            session.run([self.num_correct, self.num_total, self.auc_op], feed_dict=feed_dict)
        return num_correct, num_total, auc


    def __init__(self, num_topics, hidden_size, max_length):
        self.num_topics = num_topics
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.setup_system()


#Takes list of tuples of fields eg [(seqs0, lens0), (seqs1, lens1)...]
#returns batches of form [seqs_batch, lens_batch, ...]
def batchify(data):
    num_batches = int(np.ceil(len(data) / BATCH_SIZE))
    num_fields = len(data[0])

    batches = []
    for i in range(num_batches):
        #batch up all the different sequences of data (seqs, lens, etc)
        batch_data = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        batch = [[] for i in range(len(batch_data[0]))]
        for b in batch_data:
            for j in range(num_fields):
                batch[j].append(b[j])
        batches.append(batch)

    return batches


def train_model(model, session, data):
    seqs, lens, masks, answers, topics = data
    assert(len(seqs) == len(lens) == len(masks) == len(answers) == len(topics))
    cutoff = int(len(seqs) * TRAIN_SPLIT)
    zipped_data = list(zip(*data))
    train_data = zipped_data[:cutoff]
    test_data = zipped_data[cutoff:]

    for epoch in range(MAX_EPOCHS):
        np.random.shuffle(train_data)
        train_batches = batchify(train_data)

        print("Starting training epoch", epoch)
        for batch_num, train_batch in enumerate(train_batches):
            loss = model.train_on_batch(session, *train_batch)
            print("On batch number {}, loss is {}".format(batch_num, loss))

        print("Epoch {} complete, evaluating model...".format(epoch))
        eval_model(test_data, model, session)

def eval_model(test_data, model, session):
    np.random.shuffle(test_data)
    test_batches = batchify(test_data)
    total_correct = 0 #total number of examples we got right
    total_total = 0 #total number of examples we tried on
    total_auc = 0.0
    for test_batch in test_batches:
        num_correct, num_total, auc = model.test_on_batch(session, *test_batch)
        total_auc += num_total * auc
        total_correct += num_correct
        total_total += num_total
    print("{} test examples right out of {}, which is {} percent. Mean AUC {}\n".format(
        total_correct, total_total, 100.0*total_correct/total_total, 1.0*total_auc/total_total))


def main(_):
    print('tf version', tf.__version__)

    topics, answers, num_topics = read_assistments_data(DATA_LOC)
    full_data = load_data(topics, answers, num_topics)
    model = DKTModel(num_topics, HIDDEN_SIZE, MAX_LENGTH)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        #We need to explicitly initialize local variables to use
        #TensorFlow's AUC function for some reason...
        session.run(tf.local_variables_initializer())
        train_model(model, session, full_data)



if __name__ == "__main__":
    tf.app.run()
