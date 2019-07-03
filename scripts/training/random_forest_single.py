"""Sacred wrapper around random forest training and evaluation."""
# Since this gets wrapped by sacred, pylint will incorrectly identify variables
#   as being unused and not being given to functions.
# pylint: disable=unused-variable,no-value-for-parameter
import logging
import os

import joblib
from sacred import Experiment
from sacred.observers import MongoObserver

import scripts.helper.models as models

experiment_name = "random_forest_single"

ex = Experiment(experiment_name)
ex.observers.append(MongoObserver.create(
    url=f"mongodb+srv://{os.environ['MOORHEN_USERNAME']}:"
    f"{os.environ['MOORHEN_PASSWORD']}@moorhen-5migi.mongodb.net/",
    db_name="MY_DB"))


@ex.config  # Configuration is defined through local variables.
def cfg():
    """Config definitions for sacred"""
    representation = "bag_of_words"
    dataset = "beta/rand"
    seed = 4213
    n_estimators = 400
    max_features = 'sqrt'
    max_depth = 60


@ex.capture
def construct_save_dir(dataset, representation):
    base = os.path.join(dataset, representation, experiment_name)
    return models.create_experiment_save_dir(base)


@ex.capture
def get_data(dataset, representation):
    """Get data corresponding to representation."""
    return models.load_data(dataset, representation)


@ex.capture
def train_model(data, n_estimators, max_features, max_depth):
    """Train a series of Random Forests with the given parameters."""
    model = models.train_random_forest(data=data,
                                       n_estimators=n_estimators,
                                       max_features=max_features,
                                       max_depth=max_depth)

    return model


@ex.automain  # Using automain to enable command line integration.
def run(_run):
    """Main method that will be wrapped by sacred. Loads data, trains and prints
    out summaries."""
    save_dir = construct_save_dir()
    logging.info(f"Save directory is {save_dir}")

    logging.info(f"Loading data")
    data = get_data()  # parameters injected automatically
    logging.info(f"Loaded data")

    _run.log_scalar("X_train_size", len(data['X_train']))
    _run.log_scalar("X_val_size", len(data['X_val']))

    logging.info(f"Training model")
    model = train_model(data)
    logging.info(f"Model parameters:\n{model.get_params()}")

    logging.info(f"Saving model")
    model_filename = os.path.join(save_dir, "trained_model.joblib")
    joblib.dump(model, model_filename)

    logging.info(f"Evaluating model performance")
    short_metrics, long_metrics, plots = models.evaluate_model(model,
                                                               data,
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

    parameters_filename = os.path.join(save_dir, "parameters.json")
    models.save_to_json(model.get_params(), parameters_filename)

    logging.info(f"Saving artifacts to sacred")
    ex.add_artifact(metrics_filename)
    ex.add_artifact(parameters_filename)
    for plot_file in plots.values():
        ex.add_artifact(plot_file)

    return short_metrics['accuracy']
