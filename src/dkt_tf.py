import numpy as np
import tensorflow as tf
from collections import defaultdict
from IPython import embed
import matplotlib.pyplot as plt


DATA_LOC = './assistments.txt'
MAX_LENGTH = 100
LR = 0.01
HIDDEN_SIZE = 200

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

class DKTModel(object):

    # Takes jagged arrays of topic and answer sequences
    # Pads them into square arrays; returns that and a mask
    def load_data(self, topic_seqs, answer_seqs):
        assert(len(topic_seqs) == len(answer_seqs))
        N = len(topic_seqs)

        topics_padded = []
        answers_padded = []
        masks = []
        self.embeddings = np.zeros((N, self.max_length, self.num_topics*2))
        self.seq_lens = []

        for i in xrange(N):
            assert(len(topic_seqs[i]) == len(answer_seqs[i]))
            seq_len = len(topic_seqs[i])
            self.seq_lens.append(seq_len)
            padding = [0] * (self.max_length - seq_len)
            topics_padded.append(topic_seqs[i] + padding)
            answers_padded.append(answer_seqs[i] + padding)
            masks.append([1] * seq_len + padding)
            for j in xrange(seq_len):
                # Make the one-hot representation; for each sequence and time step,
                # 2*n_topics length vector, one-hot for topic/answer index
                one_hot_index = topic_seqs[i][j] * 2 + answer_seqs[i][j]
                self.embeddings[i][j][one_hot_index] = 1

        #padded topics and answers not actually used, could remove?
        self.topics = topics_padded
        self.answers = answers_padded
        self.masks = masks


    def add_placeholders(self):
        #self.topics_placeholder = tf.placeholder(tf.int32,(None, MAX_LENGTH))
        #self.answers_placeholder = tf.placeholder(tf.int32, (None, MAX_LENGTH))
        self.seqs_placeholder = tf.placeholder(tf.float32, (None, self.max_length, 2 * self.num_topics))
        self.seq_lens_placeholder = tf.placeholder(tf.int32, (None))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, ())
        self.lr_placeholder = tf.placeholder_with_default(LR, ())


    def build_pipeline(self):
        d = 1.0 - self.dropout_placeholder
        h = self.hidden_size
        ins = self.seqs_placeholder

        cell = tf.contrib.rnn.LSTMCell(h)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = d)

        xav_init = tf.contrib.layers.xavier_initializer()
        w = tf.get_variable("W", (h,h), tf.float32, xav_init)
        b = tf.get_variable("b", (h), tf.float32, xav_init)

        outputs, hidden_states = tf.nn.dynamic_rnn(
            cell=cell, inputs=self.seqs_placeholder,
            sequence_length=self.seq_lens_placeholder, dtype=tf.float32,
            swap_memory=True)


        #TODO: the dimensions are probably wrong (i typed this out quickly) - check them
        w2 = tf.get_variable("W2", (self.hidden_size, self.num_topics), tf.float32, xav_init)
        b2 = tf.get_variable("b2", (self.num_topics,), tf.float32, tf.constant_initializer(0.0))
        outputs_flat = tf.reshape(outputs, [-1, self.hidden_size])
        inner = tf.matmul(outputs_flat, w2) + b2

        #logit of p(correct) predicted for each time step
        self.logits = tf.reshape(inner, [-1, self.max_length, self.num_topics])

        self.probs = np.softmax(self.logits)

    def loss(self):
        #Find the cross-entropy loss between the correct answer and the
        #probability of the answer of the correct class!
        pass

    def training_op(self):
        pass

    def __init__(self, num_topics, hidden_size, max_length):
        self.num_topics = num_topics
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.add_placeholders()
        self.build_pipeline()


def train_on_batches():
    pass


def main(_):
    print 'tf version', tf.__version__

    topics, answers, num_topics = read_assistments_data(DATA_LOC)
    model = DKTModel(num_topics, HIDDEN_SIZE, MAX_LENGTH)
    model.load_data(topics, answers)

    #plt.hist(map(len,tops),bins=20, range = (0,200))
    #plt.show()
    #embed()


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())



if __name__ == "__main__":
    tf.app.run()
