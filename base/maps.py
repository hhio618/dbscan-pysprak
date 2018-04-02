from pyspark.mllib.linalg import Vectors

from base.utils import MyDBSCAN


def merge_centers(sigma, centers):
    centers = centers.collect()
    keys = [k for k, v in centers]
    clusters = MyDBSCAN() \
        .fit(centers, sigma, 1)
    return dict(zip(keys, clusters))


def find_d_min(kv):
    X = list(kv[1])
    minimum = Vectors.norm(X[0] - X[1], 2)  # lets start from somewhere
    count = len(X)
    for i in xrange(count):
        for j in xrange(i + 1, count):
            d = Vectors.norm(X[i] - X[j], 2)
            if d < minimum:
                minimum = d
    return minimum


def dbscan(kv):
    X = kv[1]
    return zip(zip([kv[0]] * len(X), MyDBSCAN().fit(X, 4, 2)), X)


def map_vector(input_line):
    lst = input_line.split(",")
    value1 = lst[0]
    value2 = Vectors.dense(map(float, lst[1:]))
    return value1, value2
