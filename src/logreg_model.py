import numpy as np
from IPython import embed
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

class LogRegModel:
    def __init__(self, num_topics):
        self.num_topics = num_topics
        self.models = {}

    def train(self, data):
        wrongs = defaultdict(list)
        rights = defaultdict(list)
        answers = defaultdict(list)

        for sequence in data:
            seq_len, masks, ans, topics = sequence
            num_right = defaultdict(int)
            num_wrong = defaultdict(int)
            for i in range(seq_len):
                topic = topics[i]
                answer = ans[i]
                wrongs[topic].append(num_wrong[topic])
                rights[topic].append(num_right[topic])
                answers[topic].append(answer)

                if answer:
                    num_right[topic] += 1
                else:
                    num_wrong[topic] += 1

        for i in range(self.num_topics):
            print i
            model = LogisticRegression()
            if len(answers[i]) == 0:
                model.coef_ = np.zeros((1,1))
                model.intercept_ = np.zeros((1))
            elif wrongs[i][-1] == 0:
                model.coef_ = np.zeros((1,1))
                model.intercept_ = np.ones((1)) * 100
            elif rights[i][-1] == 0:
                model.coef_ = np.zeros((1,1))
                model.intercept_ = np.ones((1)) * 100
            else:
                X = zip(wrongs[i], rights[i])
                Y = answers[i]
                model.fit(X,Y)
            self.models[i] = model

        embed()



