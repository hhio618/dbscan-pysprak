from collections import Counter

from base.maps import *
from pyspark import SparkConf
from pyspark import SparkContext
import random
import os

MAPS_COUNT = 3
NUM_SAMPLES = 3000

if __name__ == '__main__':
    # loading data
    raw_kvs = []
    random.seed(0)
    with open("data%sletter-recognition.data" % os.sep) as input_file:
        raw_lines = input_file.read().splitlines()
        random.shuffle(raw_lines)
        num_data = len(raw_lines)
        map_size = int(num_data / float(MAPS_COUNT))
        for i in xrange(num_data):
            part_id = i / map_size + 1
            raw_kvs.append((part_id, raw_lines[i]))

    # Create spark config
    spark_conf = SparkConf()
    spark_conf.setMaster("local")
    spark_conf.set("spark.executor.memory", "2048MB")
    spark_conf.set("spark.driver.host", "localhost")
    # initialize spark context
    spark_context = SparkContext(conf=spark_conf)
    rdd_base = spark_context.parallelize(raw_kvs[:NUM_SAMPLES]) \
        .mapValues(map_vector) \
        .map(lambda x: (x[0], [x[1]])) \
        .reduceByKey(lambda x, y: x + y) \
        .flatMap(dbscan) \
        .filter(lambda kv: kv[0][1] != -1) \
        .cache()

    # Calculating centers
    rdd_centers = rdd_base \
        .mapValues(lambda x: (x[1], 1)) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda x: x[0] / x[1])

    # Calculating D_min
    d_min = rdd_centers \
        .map(lambda x: (x[0][0], x[1])) \
        .groupByKey() \
        .map(find_d_min) \
        .reduce(min)

    # Merge clusters
    merged_centers = merge_centers(d_min / 15, rdd_centers)
    rdd_global = rdd_base \
        .map(lambda x: (merged_centers[x[0]], x[1]))
    temp = rdd_global.map(lambda x: (x[0], x[1][0])) \
        .groupByKey() \
        .mapValues(lambda v: Counter(v).most_common(1)[0][0]) \
        .collect()
    global_label = dict(temp)
    nmi = rdd_global.map(lambda x: 1 if global_label[x[0]] == x[1][0] else 0) \
              .filter(lambda x: x == 1) \
              .count() / float(NUM_SAMPLES)
    print "Number of Clusters: %d" % \
          rdd_global \
              .map(lambda x: global_label[x[0]]) \
              .distinct() \
              .count()
    print "Noises count: %d" % (NUM_SAMPLES - rdd_base.count())
    print "NMI is %f" % nmi
