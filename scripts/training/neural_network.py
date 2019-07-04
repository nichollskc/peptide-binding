import logging

import tensorflow as tf
import numpy as np
import os


def compute_accuracy(computed_probs, labels):
    predictions = np.where(computed_probs > 0.5, 1, 0).ravel()
    accuracy = np.mean(predictions == labels)
    return accuracy


dataset = "beta/clust"
representation = "bag_of_words"
logging.info(f"Loading data from dataset {dataset}, representation {representation}")
x_train = np.load(f"datasets/{dataset}/training/data_{representation}.npy")
y_train = np.load(f"datasets/{dataset}/training/labels.npy")

x_valid = np.load(f"datasets/{dataset}/validation/data_{representation}.npy")
y_valid = np.load(f"datasets/{dataset}/validation/labels.npy")

tf.reset_default_graph()
input_layer = tf.placeholder(tf.float32, shape=(None, 42), name="x")

training = tf.placeholder(tf.bool, name="training")

name = "exp5"
dense = tf.layers.dense(input_layer, 10, name="dense", activation=tf.nn.relu)
second = tf.layers.dense(dense, 10, name="second", activation=tf.nn.relu)
dropped = tf.layers.dropout(second, rate=0.5, training=training, name="dropout")
logits = tf.layers.dense(dropped, 1, name="logits")
probs = tf.nn.sigmoid(logits, name="probs")

labels = tf.placeholder(tf.float32, shape=(None, 1), name="labels")
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name="loss")
print("Loss shape: {}".format(loss.shape.as_list()))
print("Logits shape: {}".format(logits.shape.as_list()))

loss_summary = tf.summary.scalar('loss_summary', tf.reduce_mean(loss))
merged = tf.summary.merge_all()

train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

sess = tf.Session()
tensorboard_path = 'logs/tensorboard'
tensorboard_file_name = os.path.join(tensorboard_path, name)

train_writer = tf.summary.FileWriter(tensorboard_file_name, sess.graph)

sess.run(tf.global_variables_initializer())
global_step = 0
for i in range(1000):
    j = 0
    mb_size = 100
    losses = []
    accuracies = []
    while j + mb_size < len(x_train):
        _ , computed_loss, computed_probs, summary = sess.run([train_op, loss, probs, merged],
                                    feed_dict={input_layer: x_train[j:j + mb_size, :],
                                               labels: np.expand_dims(y_train[j:j + mb_size], axis=1),
                                               training: True,})
        train_writer.add_summary(summary, global_step)
        accuracy = compute_accuracy(computed_probs, y_train[j:j + mb_size])
        accuracies.append(accuracy)
        losses.append(computed_loss)
        j += mb_size
        global_step += 1
    print("Epoch {}, mean training loss: {:.3f}, mean training accuracy: {:.3f}".format(
          i, np.mean(losses), np.mean(accuracies)))

    computed_probs = sess.run(probs, feed_dict={
        input_layer: x_train,
        training: False,
    })
    train_accuracy = compute_accuracy(computed_probs, y_train)
    print("Train accuracy: {:.3f}".format(train_accuracy))

    computed_probs = sess.run(probs, feed_dict={
        input_layer: x_valid,
        training: False,
    })
    valid_accuracy = compute_accuracy(computed_probs, y_valid)
    print("Valid accuracy: {:.3f}".format(valid_accuracy))
