import numpy as np
import tensorflow as tf
from collections import defaultdict
from IPython import embed
import matplotlib.pyplot as plt


DATA_LOC = './assistments.txt'
MAX_LENGTH = 100
LR = 0.01
HIDDEN_SIZE = 200
BATCH_SIZE = 32
MAX_EPOCHS = 10
DROPOUT = 0.3
TRAIN_SPLIT = 0.7

def read_assistments_data(loc):
    with open(loc) as f:
        data = [map(int,x.strip().split()) for x in f.readlines()]

    topic_seqs = defaultdict(list)
    answer_seqs = defaultdict(list)
    topics = set()

    for d in data:
        student_id = d[0]
        if(len(topic_seqs[student_id])) >= MAX_LENGTH: continue
        topic_seqs[student_id].append(d[1])
        answer_seqs[student_id].append(d[2])
        topics.add(d[1])

    #topics may not be consecutively numbered but assume they are
    return topic_seqs.values(), answer_seqs.values(), max(topics) + 1


# Takes jagged arrays of topic and answer sequences
# Pads them into square arrays; returns that and a mask
def load_data(topic_seqs, answer_seqs, num_topics):
    assert(len(topic_seqs) == len(answer_seqs))
    N = len(topic_seqs)

    #topics_padded = []
    answers_padded = []
    masks = []
    embeddings = np.zeros((N, MAX_LENGTH, num_topics*2))
    seq_lens = []

    for i in xrange(N):
        assert(len(topic_seqs[i]) == len(answer_seqs[i]))
        seq_len = len(topic_seqs[i])
        seq_lens.append(seq_len)
        padding = [0] * (MAX_LENGTH - seq_len)
        #topics_padded.append(topic_seqs[i] + padding)
        answers_padded.append(answer_seqs[i] + padding)
        masks.append([1] * seq_len + padding)
        for j in xrange(seq_len):
            # Make the one-hot representation; for each sequence and time step,
            # 2*n_topics length vector, one-hot for topic/answer index
            one_hot_index = topic_seqs[i][j] * 2 + answer_seqs[i][j]
            embeddings[i][j][one_hot_index] = 1

    #padded topics and answers not actually used, could remove?
    #self.topics = topics_padded
    answers = answers_padded
    return embeddings, seq_lens, masks, answers

def train_data_part(seqs, lens, masks, answers):
    assert(len(seqs) == len(lens) == len(masks) == len(answers))
    N = len(seqs)
    cutoff = int(N * TRAIN_SPLIT)
    return seqs[:cutoff], lens[:cutoff], masks[:cutoff], answers[:cutoff]

def test_data_part(seqs, lens, masks, answers):
    assert(len(seqs) == len(lens) == len(masks) == len(answers))
    N = len(seqs)
    cutoff = int(N * TRAIN_SPLIT)
    return seqs[cutoff:], lens[cutoff:], masks[cutoff:], answers[:cutoff]



class DKTModel(object):


    def add_placeholders(self):
        #self.topics_placeholder = tf.placeholder(tf.int32,(None, MAX_LENGTH))
        self.answers_placeholder = tf.placeholder(tf.int32, (None, MAX_LENGTH))
        self.seqs_placeholder = tf.placeholder(tf.float32, (None, self.max_length, 2 * self.num_topics))
        self.seq_lens_placeholder = tf.placeholder(tf.int32, (None))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder_with_default(DROPOUT, ())
        self.lr_placeholder = tf.placeholder_with_default(LR, ())


    def data_pipeline(self):
        d = 1.0 - self.dropout_placeholder
        h = self.hidden_size
        ins = self.seqs_placeholder

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

        #logit of p(correct) predicted for each time step
        logits = tf.reshape(inner, [-1, self.max_length, self.num_topics])
        #Probabilities of getting each class at each time correct, for reference
        self.probs = tf.nn.softmax(logits)
        return logits


    def find_loss(self, logits, labels):
        #Find the cross-entropy loss between the correct answer and the
        #probability of the answer of the correct class!
        ce_wl = tf.nn.sparse_softmax_cross_entropy_with_logits
        losses = ce_wl(logits=logits, labels=labels)
        masked_losses = tf.boolean_mask(losses, self.mask_placeholder)

        #sum loss over all indices, treating each time-step as a separate
        #data point
        total_loss = tf.reduce_sum(masked_losses)
        return total_loss


    def setup_system(self):
        self.add_placeholders()

        with tf.variable_scope("dkt"):
            self.logits = self.data_pipeline()

        self.loss = self.find_loss(self.logits, self.answers_placeholder)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


    def train_on_batch(self, session, seqs_batch, lens_batch, masks_batch, answers_batch):
        feed_dict = {self.seqs_placeholder: seqs_batch,
                     self.seq_lens_placeholder: lens_batch,
                     self.mask_placeholder: masks_batch,
                     self.answers_placeholder: answers_batch}
        _, loss = session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss


    def __init__(self, num_topics, hidden_size, max_length):
        self.num_topics = num_topics
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.setup_system()


#Get the results out; don't train the model
def run_model(seqs, lengths, masks):
    pass

def batchify(data):
    num_batches = int(np.ceil(len(data) / BATCH_SIZE))
    batches = []
    for i in range(num_batches):
        #batch up all the different sequences of data (seqs, lens, etc)
        batches.append([d[i*BATCH_SIZE:(i+1)*BATCH_SIZE] for d in data])
    return batches


def train_model(model, session, data):
    train_data = train_data_part(*data)[:]
    test_data = test_data_part(*data)[:]

    for epoch in range(MAX_EPOCHS):
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        train_batches = batchify(train_data)
        test_batches = batchify(test_data)

        print "Starting training epoch", epoch
        for batch_num in range(len(train_batches)):
            train_batch = train_batches[batch_num]
            loss = model.train_on_batch(session, *train_batch)
            print "On batch number {}, loss is {}".format(batch_num, loss)

        #TODO: This isn't right, fix it
        #error_rate = model.error_rate(test_batches)
        #print "Error rate on test set after this epoch: ", error_rate


def main(_):
    print 'tf version', tf.__version__

    topics, answers, num_topics = read_assistments_data(DATA_LOC)
    full_data = load_data(topics, answers, num_topics)
    #embeddings, seq_lengths, masks, answers = load_data(topics, answers, num_topics)
    #train_data = train_data_part(embeddings, seq_lengths, masks, answers)

    model = DKTModel(num_topics, HIDDEN_SIZE, MAX_LENGTH)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        train_model(model, session, full_data)



if __name__ == "__main__":
    tf.app.run()
