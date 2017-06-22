import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


np.load('alex_stats.npy')
stats = np.load('alex_stats.npy')

# stats[1] is results on eval DKT simulator (trained on second half of data)
# stats[0] is results on train DKT simulator (trained on first half of data)
eval_stats = stats[1]

#delta-p, per student, per topic, for the random policy
random_scores = eval_stats[0]
#delta-p, per student, per topic, for the policy-gradient trained policy
trained_scores = eval_stats[1]


#Total improvement (summed across all topics), per student, for the random and trained policies
#random_vhats = np.sum(random_scores,axis=1)
#trained_vhats = np.sum(trained_scores,axis=1)
#
#print len(random_vhats), len(trained_vhats)
#
#rand_mean = np.round(np.mean(random_vhats), 2)
#rand_std = np.round(np.std(random_vhats), 2)
#
#trained_mean = np.round(np.mean(trained_vhats), 2)
#trained_std = np.round(np.std(trained_vhats), 2)
#
#plt.hist(random_vhats, label='Random policy (mean {}; stdev {}; 256 students)'.format(rand_mean, rand_std), alpha=0.5)
#plt.hist(trained_vhats, label='Trained policy (mean {}; stdev {}; 256 students)'.format(trained_mean, trained_std), alpha=0.5)
#
##sns.distplot(random_vhats, label='Random policy')
##sns.distplot(trained_vhats, label='Trained policy')
#
#plt.xlabel('Mean delta-p improvement per student across all topics', fontsize=16)
#plt.ylabel('Number of students', fontsize=16)
#plt.legend(fontsize=16, loc=2)
#plt.title('Distribution of student improvement over 100 time steps\n '
#          'Our policy vs random policy, evaluated on DKT model trained on eval set', fontsize=20 )
#plt.show()



#Total improvement (summed across all students), per topic, for the random and trained policies
random_vhats = np.mean(random_scores,axis=0)
trained_vhats = np.mean(trained_scores,axis=0)

print len(random_vhats), len(trained_vhats)
print np.sum(random_vhats), np.sum(trained_vhats)
print np.mean(random_vhats), np.mean(trained_vhats)
#print np.sum(random_scores), np.sum(trained_scores)
#print random_scores.shape, trained_scores.shape


rand_mean = np.round(np.mean(random_vhats), 2)
rand_std = np.round(np.std(random_vhats), 2)

trained_mean = np.round(np.mean(trained_vhats), 2)
trained_std = np.round(np.std(trained_vhats), 2)

plt.hist(random_vhats, label='Random policy (mean {}; stdev {}; 124 topics)'.format(rand_mean, rand_std), alpha=0.5)
plt.hist(trained_vhats, label='Trained policy (mean {}; stdev {}; 124 topics)'.format(trained_mean, trained_std), alpha=0.5)

#sns.distplot(random_vhats, label='Random policy')
#sns.distplot(trained_vhats, label='Trained policy')

plt.xlabel('Mean delta-p improvement per topic across all students', fontsize=16)
plt.ylabel('Number of topics', fontsize=16)
plt.legend(fontsize=16, loc=2)
plt.title('Distribution of per-topic improvement over 100 time steps\n '
          'Our policy vs random policy, evaluated on DKT model trained on eval set', fontsize=20 )
plt.show()
