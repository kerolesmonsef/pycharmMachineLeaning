import tensorflow.compat.v1 as tf

import pickle

import numpy as np

tf.disable_v2_behavior()

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2
batch_size = 100

X_train, y_train, X_test, y_test = pickle.load(open('sentiment_set.pickle', 'rb'))

x = tf.placeholder(dtype=tf.float32, shape=[None, len(X_train[0])])
y = tf.placeholder(dtype=tf.int32)
y_onehot = tf.one_hot(indices=y, depth=n_classes)


def my_neural_network_model():
    hidden_layer_1_weights = {
        'weight': tf.Variable(tf.random.normal([len(X_train[0]), n_nodes_hl1]), name="weight_h_1"),
        'baises': tf.Variable(tf.random.normal([n_nodes_hl1]), name="baises_h_1"),
    }

    hidden_layer_2_weights = {
        'weight': tf.Variable(tf.random.normal([n_nodes_hl1, n_nodes_hl2]), name="weight_h_2"),
        'baises': tf.Variable(tf.random.normal([n_nodes_hl2]), name="baises_h_2"),
    }

    hidden_layer_3_weights = {
        'weight': tf.Variable(tf.random.normal([n_nodes_hl2, n_nodes_hl3]), name="weight_h_3"),
        'baises': tf.Variable(tf.random.normal([n_nodes_hl3]), name="baises_h_3"),
    }

    output_layer_weights = {
        'weight': tf.Variable(tf.random.normal([n_nodes_hl3, n_classes]), name="output_w"),
        'baises': tf.Variable(tf.random.normal([n_classes]), name="output_b"),
    }

    # y = X * weight + b

    # [1 * 784 ] * [784 * 500] + [1 * 500 ] => [ 1 * 500 ]
    l1 = tf.matmul(x, hidden_layer_1_weights['weight']) + hidden_layer_1_weights['baises']
    # l1 = tf.nn.relu(l1)

    # [1 * 500 ] * [500 * 500] + [1 * 500 ] => [ 1 * 500 ]
    l2 = tf.matmul(l1, hidden_layer_2_weights['weight']) + hidden_layer_2_weights['baises']
    l2 = tf.nn.relu(l2)

    # [1 * 500 ] * [500 * 500] + [1 * 500 ] => [ 1 * 500 ]
    l3 = tf.matmul(l2, hidden_layer_3_weights['weight']) + hidden_layer_3_weights['baises']
    l3 = tf.nn.relu(l3)

    # [1 * 500 ] * [500 * 10] + [1 * 10 ] => [ 1 * 10 ]
    output = tf.matmul(l3, output_layer_weights['weight']) + output_layer_weights['baises']

    return output


output = my_neural_network_model()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_onehot))
# cost = tf.losses.softmax_cross_entropy(logits=output, onehot_labels=y_onehot)
optimizer = tf.train.AdamOptimizer().minimize(cost)

predictions = {
    'classes': tf.argmax(output, axis=1, name='predicted_classes'),
    'probabilities': tf.nn.softmax(output, name='softmax_tensor')
}

hm_epochs = 10
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(hm_epochs):
    epoch_loss = 0
    for i in range(0, len(X_train), batch_size):
        epoch_x = X_train[i:i + batch_size]
        epoch_y = y_train[i:i + batch_size]
        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
        epoch_loss += c

    print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

y_pred = sess.run(predictions['classes'], feed_dict={x: X_test})
print('Test Accuracy: %.2f%%' % (100 * np.sum(y_pred == y_test) / len(y_test)))
