"""Sacred wrapper around random forest training and evaluation."""
# Since this gets wrapped by sacred, pylint will incorrectly identify variables
#   as being unused and not being given to functions.
# pylint: disable=unused-variable,no-value-for-parameter
import logging
import os
import tensorflow as tf
import numpy as np

import joblib
from sacred import Experiment
from sacred.observers import MongoObserver

import peptidebinding.helper.models as models

experiment_name = "neural_network"

ex = Experiment(experiment_name)
ex.observers.append(MongoObserver.create(
    url=f"mongodb+srv://{os.environ['MOORHEN_USERNAME']}:"
        f"{os.environ['MOORHEN_PASSWORD']}@moorhen-5migi.mongodb.net/",
    db_name="MY_DB"))


def compute_accuracy(computed_probs, labels):
    predictions = np.where(computed_probs > 0.5, 1, 0).ravel()
    accuracy = np.mean(predictions == labels)
    return accuracy


@ex.config  # Configuration is defined through local variables.
def cfg():
    """Config definitions for sacred"""
    representation = "bag_of_words"
    dataset = "beta/clust"
    learning_rate = 1e-3
    dropout = 0.3
    units_layer1 = 15
    units_layer2 = 10
    seed = 1342
    epochs = 500
    mb_size = 100
    regularisation_weight = 0.1


@ex.capture
def construct_save_dir(dataset, representation):
    save_dir = os.path.join(models.MODELS_DIR, dataset, representation, experiment_name)
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        # directory already exists
        pass
    return save_dir

@ex.capture
def get_data(dataset, representation):
    """Get data corresponding to representation."""
    data = models.load_data(dataset, representation)
    return data['X_train'].toarray(), data['y_train'], data['X_val'].toarray(), data['y_val']


@ex.capture
def train_model(x_train, y_train, x_valid, y_valid, regularisation_weight, learning_rate, mb_size, epochs, units_layer1, units_layer2, dropout, seed, _run):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    input_layer = tf.placeholder(tf.float32, shape=(None, x_train.shape[1]), name="x")

    training = tf.placeholder(tf.bool, name="training")

    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    name = f"2_layer_lr_{learning_rate}_ul1_{units_layer1}_ul2_{units_layer2}_dropout_{dropout}_regularisation_{regularisation_weight}"
    dense = tf.layers.dense(input_layer, units_layer1, name="dense", activation=tf.nn.relu)
    second = tf.layers.dense(dense, units_layer2, name="second", activation=tf.nn.relu)
    dropped = tf.layers.dropout(second, rate=dropout, training=training, name="dropout")
    logits = tf.layers.dense(dropped, 1, name="logits", kernel_regularizer=regularizer)
    probs = tf.nn.sigmoid(logits, name="probs")

    labels = tf.placeholder(tf.float32, shape=(None, 1), name="labels")
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name="loss")

    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    logging.info("Loss shape: {}".format(loss.shape.as_list()))
    logging.info("Logits shape: {}".format(logits.shape.as_list()))

    loss_summary = tf.summary.scalar('loss_summary', tf.reduce_mean(loss))
    merged = tf.summary.merge_all()

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session()
    tensorboard_path = 'logs/tensorboard'
    tensorboard_file_name = os.path.join(tensorboard_path, name)

    train_writer = tf.summary.FileWriter(tensorboard_file_name, sess.graph)

    sess.run(tf.global_variables_initializer())
    global_step = 0
    for i in range(epochs):
        j = 0
        losses = []
        accuracies = []
        while j < x_train.shape[0]:
            _, computed_loss, computed_probs, summary = sess.run([train_op, loss, probs, merged],
                                                                 feed_dict={input_layer: x_train[
                                                                                         j:j + mb_size,
                                                                                         :],
                                                                            labels: np.expand_dims(
                                                                                y_train[
                                                                                j:j + mb_size],
                                                                                axis=1),
                                                                            training: True, })
            train_writer.add_summary(summary, global_step)
            accuracy = compute_accuracy(computed_probs, y_train[j:j + mb_size])
            accuracies.append(accuracy)
            losses.append(computed_loss)
            j += mb_size
            global_step += 1
        logging.info("Epoch {}, mean training loss: {:.3f}, mean training accuracy: {:.3f}".format(
            i, np.mean(np.vstack(losses)), np.mean(accuracies)))

        j = 0
        y_train_probs_batched = []
        while j < x_train.shape[0]:
            y_train_probs_batched.append(sess.run(probs, feed_dict={
                input_layer: x_train[j:j + mb_size, :],
                training: False,
            }))
            j += mb_size
        y_train_probs = np.vstack(y_train_probs_batched)
        y_train_pred = np.where(y_train_probs > 0.5, 1, 0).ravel()
        train_accuracy = compute_accuracy(y_train_probs, y_train)
        logging.info("Train accuracy: {:.3f}".format(train_accuracy))

        j = 0
        y_valid_probs_batched = []
        while j < x_valid.shape[0]:
            y_valid_probs_batched.append(sess.run(probs, feed_dict={
                input_layer: x_valid[j:j + mb_size, :],
                training: False,
            }))
            j += mb_size
        y_valid_probs = np.vstack(y_valid_probs_batched)
        y_valid_pred = np.where(y_valid_probs > 0.5, 1, 0).ravel()
        valid_accuracy = compute_accuracy(y_valid_probs, y_valid)
        logging.info("Valid accuracy: {:.3f}".format(valid_accuracy))

        _run.log_scalar("learning_train_accuracy", train_accuracy)
        _run.log_scalar("learning_validation_accuracy", valid_accuracy)
    return y_valid_pred, y_valid_probs, y_train_pred, y_train_probs


@ex.automain  # Using automain to enable command line integration.
def run(_run):
    """Main method that will be wrapped by sacred. Loads data, trains and prints
    out summaries."""
    save_dir = construct_save_dir()
    logging.info(f"Save directory is {save_dir}")

    logging.info(f"Loading data")
    x_train, y_train, x_valid, y_valid = get_data()   # parameters injected automatically
    logging.info(f"Loaded data")

    _run.log_scalar("X_train_size", x_train.shape[0])
    _run.log_scalar("X_val_size", x_valid.shape[0])

    logging.info(f"Training model")
    y_valid_pred, y_valid_probs, y_train_pred, y_train_probs = train_model(x_train,
                                                                           y_train,
                                                                           x_valid,
                                                                           y_valid)

    logging.info(f"Evaluating model performance")
    short_metrics, long_metrics, plots = models.evaluate_predictions(y_valid_pred,
                                                                     y_valid_probs,
                                                                     y_valid,
                                                                     y_train_pred,
                                                                     y_train_probs,
                                                                     y_train,
                                                                     save_dir)

    logging.info(f"Saving metrics to sacred")
    # Log the single number metrics using sacred
    for key, value in short_metrics.items():
        _run.log_scalar(key, value)

    logging.info(f"Saving metrics to file")
    # Combine the two metrics dictionaries into one and save as a json file
    full_metrics = short_metrics.copy()
    full_metrics.update(long_metrics)

    metrics_filename = os.path.join(save_dir, "metrics.json")
    models.save_to_json(full_metrics, metrics_filename)

    logging.info(f"Saving artifacts to sacred")
    ex.add_artifact(metrics_filename)
    for plot_file in plots.values():
        ex.add_artifact(plot_file)

    return short_metrics['accuracy']
