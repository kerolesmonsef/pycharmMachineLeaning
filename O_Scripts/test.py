import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

with tf.name_scope('scalar_set_one') as scope:
    tf_constant_one = tf.constant(10, name="ten")
    tf_constant_two = tf.constant(20, name="twenty")
    scalar_sum_one = tf.add(tf_constant_one, tf_constant_two, name="scalar_ten_plus_twenty")

with tf.name_scope('scalar_set_two') as scope:
    tf_constant_three = tf.constant(30, name="thirty")
    tf_constant_four = tf.constant(40, name="fourty")
    scalar_sum_two = tf.add(tf_constant_three, tf_constant_four, name="scalar_thirty_plus_fourty")

scalar_sum_sum = tf.add(scalar_sum_one, scalar_sum_two)
sess = tf.Session()
tf_tensorboard_writer = tf.summary.FileWriter('./graphs', sess.graph)
tf_tensorboard_writer.close()
sess.close()
