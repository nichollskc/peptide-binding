"""Sacred wrapper around logistic regression training and evaluation."""
# Since this gets wrapped by sacred, pylint will incorrectly identify variables
#   as being unused and not being given to functions.
# pylint: disable=unused-variable,no-value-for-parameter
import os

import joblib
import numpy as np
from sacred import Experiment
from sacred.observers import MongoObserver

import scripts.helper.models as models

experiment_name = "logistic_regression"

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
    lr_params = {
        'alpha': np.geomspace(1e-4, 1, 10),  # Regularisation strength
        'l1_ratio': np.linspace(0, 1, 11),  # Ratio between L1 and L2 regularisation
                                            # l1_ratio = 0 => L2 penalty
                                            # l1_ratio = 1 => L1 penalty,
    }
    num_param_sets = 15
    num_folds = 10


@ex.capture
def get_data(dataset, representation):
    """Get data corresponding to representation."""
    return models.load_data(dataset, representation)


@ex.capture
def train_model_random_search(data, lr_params, num_folds, num_param_sets):
    """Train a series of models, testing out random sets of parameters
    from lr_params. Will use cross-validation with num_folds.

    Prints out summary information from the search and returns the estimator
    which had best cross-validation score."""
    search = models.random_search_logistic_regression(data=data,
                                                      param_dist=lr_params,
                                                      num_folds=num_folds,
                                                      num_param_sets=num_param_sets)

    models.summarise_search(search, num_results=num_param_sets, full_print=True)

    model = search.best_estimator_
    model.fit(data['X_train'], data['y_train'])
    return model


@ex.automain  # Using automain to enable command line integration.
def run(_run):
    """Main method that will be wrapped by sacred. Loads data, trains and prints
    out summaries."""
    save_dir = models.create_experiment_save_dir(experiment_name)

    data = get_data()  # parameters injected automatically

    _run.log_scalar("X_train_size", data['X_train'].shape[0])
    _run.log_scalar("X_val_size", data['X_val'].shape[0])
    model = train_model_random_search(data)

    model_filename = os.path.join(save_dir, "trained_model.joblib")
    joblib.dump(model, model_filename)

    short_metrics, long_metrics, plots = models.evaluate_model(model,
                                                               data,
                                                               save_dir)

    # Log the single number metrics using sacred
    for key, value in short_metrics.items():
        _run.log_scalar(key, value)

    # Combine the two metrics dictionaries into one and save as a json file
    full_metrics = short_metrics.copy()
    full_metrics.update(long_metrics)

    metrics_filename = os.path.join(save_dir, "metrics.json")
    models.save_to_json(full_metrics, metrics_filename)

    ex.add_artifact(metrics_filename)
    for plot_file in plots.values():
        ex.add_artifact(plot_file)

    return short_metrics['accuracy']
