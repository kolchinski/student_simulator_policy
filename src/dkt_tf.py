from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import defaultdict
from IPython import embed
import matplotlib.pyplot as plt


DATA_LOC = './assistments.txt'
MAX_LENGTH = 100
LR = 0.0005
HIDDEN_SIZE = 200
BATCH_SIZE = 32
MAX_EPOCHS = 10
DROPOUT = 0.3
TRAIN_SPLIT = 0.7
EMBEDDING_SIZE = 25

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
    embeddings = np.zeros((N, MAX_LENGTH + 1, num_topics*2))
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
        for j in range(seq_len):
            # Make the one-hot representation; for each sequence and time step,
            # 2*n_topics length vector, one-hot for topic/answer index
            one_hot_index = topic_seqs[i][j] * 2 + answer_seqs[i][j]
            embeddings[i][j + 1][one_hot_index] = 1

    #padded topics and answers not actually used, could remove?
    topics = topics_padded
    answers = answers_padded
    #return embeddings, seq_lens, masks, answers, topics
    return seq_lens, masks, answers, topics

class DKTModel(object):


    def add_placeholders(self):
        self.topics_placeholder = tf.placeholder(tf.int32,(None, MAX_LENGTH))
        self.answers_placeholder = tf.placeholder(tf.int32, (None, MAX_LENGTH))
        #self.seqs_placeholder = tf.placeholder(tf.float32, (None, self.max_length, 2 * self.num_topics))
        self.seq_lens_placeholder = tf.placeholder(tf.int32, (None))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder_with_default(DROPOUT, ())
        self.lr_placeholder = tf.placeholder_with_default(LR, ())


    def data_pipeline(self):
        xav_init = tf.contrib.layers.xavier_initializer()
        d = 1.0 - self.dropout_placeholder
        h = self.hidden_size

        embeddings = tf.Variable(
            tf.random_uniform([self.num_topics * 2, EMBEDDING_SIZE], -1.0, 1.0))

        #Can use this instead of seqs_placeholder
        #obs_seqs = tf.one_hot(self.topics_placeholder * 2 + self.answers_placeholder, 2*self.num_topics)
        #batch_size = tf.shape(obs_seqs)[0]
        #self.seqs = tf.concat([tf.zeros((batch_size, 1, 2*self.num_topics)), obs_seqs], 1)
        init_state1 = tf.Variable(tf.random_uniform([BATCH_SIZE, self.hidden_size], -1.0, 1.0))
        init_state2 = tf.Variable(tf.random_uniform([BATCH_SIZE, self.hidden_size], -1.0, 1.0))
        batch_size = tf.shape(self.topics_placeholder)[0]

        # add start token,
        topics_wo_start = self.topics_placeholder + self.num_topics * self.answers_placeholder
        all_topics = tf.concat((tf.zeros((batch_size, 1), dtype=tf.int32), topics_wo_start), axis=1)

        topic_seqs = tf.nn.embedding_lookup(embeddings, all_topics)
        # answers_embedding = tf.to_float(tf.reshape(self.answers_placeholder, (-1, MAX_LENGTH, 1)))
        # seqs_with_answers = tf.concat([topic_seqs, answers_embedding], 2)
        # self.seqs = tf.concat([tf.zeros((batch_size, 1, EMBEDDING_SIZE + 1)), seqs_with_answers], 1)
        self.seqs = topic_seqs

        with tf.variable_scope('lstm1'):
            cell = tf.contrib.rnn.LSTMCell(h)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=d)
            outputs1, hidden_states = tf.nn.dynamic_rnn(# initial_state=init_state1,
                cell=cell, inputs=self.seqs,
                sequence_length=self.seq_lens_placeholder + 1, dtype=tf.float32,
                swap_memory=True)

        with tf.variable_scope('lstm2'):
            cell2 = tf.contrib.rnn.LSTMCell(h)
            cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=d)

            outputs, hidden_states = tf.nn.dynamic_rnn(# initial_state=init_state2,
                cell=cell2, inputs=outputs1,
                sequence_length=self.seq_lens_placeholder + 1, dtype=tf.float32,
                swap_memory=True)

        w = tf.get_variable("W", (self.hidden_size, self.num_topics), tf.float32, xav_init)
        b = tf.get_variable("b", (self.num_topics,), tf.float32, tf.constant_initializer(0.0))
        outputs_flat = tf.reshape(outputs, [-1, self.hidden_size])
        inner = tf.matmul(outputs_flat, w) + b

        # p(correct) predicted for each time step, for each topic class
        self.all_probs = tf.sigmoid(tf.reshape(inner, [-1, self.max_length + 1, self.num_topics]))

        #Slice up the probabilities so that the last ones (for the *next*
        #questions encountered) are in a separate op
        # self.post_probs = tf.slice(self.all_probs, [0,self.seq_lens_placeholder - 1,0], [-1, -1, -1])
        self.post_probs = self.all_probs[:, self.seq_lens_placeholder[0]] # assumes all seq_lens are equal
        self.probs = tf.slice(self.all_probs, [0,0,0], [-1, self.max_length, -1])

        self.v_hats = tf.reduce_sum(self.probs, 2)

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


    def train_on_batch(self, session, lens_batch, masks_batch, answers_batch, topics_batch):
        feed_dict = {#self.seqs_placeholder: seqs_batch,
                     self.seq_lens_placeholder: lens_batch,
                     self.mask_placeholder: masks_batch,
                     self.answers_placeholder: answers_batch,
                     self.topics_placeholder: topics_batch}
        _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def test_on_batch(self, session, lens_batch, masks_batch, answers_batch, topics_batch):
        feed_dict = {#self.seqs_placeholder: seqs_batch,
                     self.seq_lens_placeholder: lens_batch,
                     self.mask_placeholder: masks_batch,
                     self.answers_placeholder: answers_batch,
                     self.topics_placeholder: topics_batch}
        num_correct, num_total, auc, v_hats = \
            session.run([self.num_correct, self.num_total, self.auc_op, self.v_hats], feed_dict=feed_dict)
        return num_correct, num_total, auc, v_hats

    # Returns array of predicted next probabilities of correctness
    # Shape 1 x batch size x 1 x num_topics
    def next_probs(self, session, lens_batch, masks_batch, answers_batch, topics_batch):
        feed_dict = {
            self.seq_lens_placeholder: lens_batch,
            self.mask_placeholder: masks_batch,
            self.answers_placeholder: answers_batch,
            self.topics_placeholder: topics_batch}
        post_probs, = session.run([self.post_probs], feed_dict=feed_dict)
        return post_probs

    def next_probs_with_topical(self, session, lens_batch, masks_batch, answers_batch, topics_batch):
        feed_dict = {
            self.seq_lens_placeholder: lens_batch,
            self.mask_placeholder: masks_batch,
            self.answers_placeholder: answers_batch,
            self.topics_placeholder: topics_batch}
        post_probs, topical_probs = session.run([self.post_probs, self.topical_probs], feed_dict=feed_dict)
        return post_probs, topical_probs

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


CKPT_FILE = './models.ckpt'
#Trains 2 DKT models, one on each half of the data
def train_paired_models(session, data, num_topics):

    with tf.variable_scope("model1"):
        model1 = DKTModel(num_topics, HIDDEN_SIZE, MAX_LENGTH)
    with tf.variable_scope("model2"):
        model2 = DKTModel(num_topics, HIDDEN_SIZE, MAX_LENGTH)

    session.run(tf.local_variables_initializer())

    lens, masks, answers, topics = data
    assert(len(lens) == len(masks) == len(answers) == len(topics))
    cutoff = int(len(lens) * 0.5)
    zipped_data = list(zip(*data))
    first_data = zipped_data[:cutoff]
    second_data = zipped_data[cutoff:]

    saver = tf.train.Saver()
    try:
        saver.restore(session, CKPT_FILE)
        return model1, model2, second_data
    except:
        pass

    session.run(tf.global_variables_initializer())

    def train_model_epoch(model, train_data, dev_data, epoch_num, ):
        np.random.shuffle(train_data)
        train_batches = batchify(train_data)

        print("starting training epoch", epoch_num)
        for batch_num, train_batch in enumerate(train_batches):
            loss = model.train_on_batch(session, *train_batch)
            if batch_num % 10 == 0:
                print("On batch number {}, loss is {}".format(batch_num, loss))

        print("Epoch {} complete, evaluating model...".format(epoch_num))
        eval_model(dev_data, model, session)

    for epoch in range(3):
        print("First Model")
        train_model_epoch(model1, first_data, second_data, epoch)

    for epoch in range(3):
        print("Second Model")
        train_model_epoch(model2, second_data, first_data, epoch)


    saver.save(session, CKPT_FILE)

    #return model1, first_seq_lens, first_answers, first_topics, first_masks, first_v_hats, \
    #    model2, second_seq_lens, second_answers, second_topics, second_masks, second_v_hats,
    return model1, model2, second_data


def test_paired_models(session, data, model1, model2):
    zipped_data = list(zip(*data))
    cutoff = int(len(data[0]) * 0.5)
    first_data = zipped_data[:cutoff]
    second_data = zipped_data[cutoff:]

    print("Evaluating first model")
    eval_model(second_data, model1, session)
    print("Evaluating second model")
    eval_model(first_data, model2, session)

def train_model(model, session, data):
    lens, masks, answers, topics = data
    assert(len(lens) == len(masks) == len(answers) == len(topics))
    cutoff = int(len(lens) * TRAIN_SPLIT)
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
        num_correct, num_total, auc, v_hats = model.test_on_batch(session, *test_batch)
        total_auc += num_total * auc
        total_correct += num_correct
        total_total += num_total
    print("{} test examples right out of {}, which is {} percent. Mean AUC {}\n".format(
        total_correct, total_total, 100.0*total_correct/total_total, 1.0*total_auc/total_total))


def get_paired_models(session, return_second_data = False):
    topics, answers, num_topics = read_assistments_data(DATA_LOC)
    full_data = load_data(topics, answers, num_topics)

    model1, model2, second_data = train_paired_models(session, full_data, num_topics)
    #test_paired_models(session, full_data, model1, model2)

    if return_second_data: return model1, model2, second_data
    return model1, model2

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
        #model1, model2 = train_paired_models(session, full_data, num_topics)
        #test_paired_models(session, full_data, model1, model2)
        #embed()



if __name__ == "__main__":
    tf.app.run()
