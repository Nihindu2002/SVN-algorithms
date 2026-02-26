import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import metrics

digits =load_digits()

#all of the feature parts
data =scale(digits.data)

y = digits.target

#amount of centroids to make
k = len(np.unique(y))

#get amount of intances(amount of numbers that we have)
samples, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.n_iter_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

#implement classifier
clf = KMeans(n_clusters=k, init ="random", n_init=10)

bench_k_means(clf,"1",data)