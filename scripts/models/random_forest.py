"""Sacred wrapper around random forest training and evaluation."""
# Since this gets wrapped by sacred, pylint will incorrectly identify variables
#   as being unused and not being given to functions.
# pylint: disable=unused-variable,no-value-for-parameter
import json
import os

from sacred import Experiment
from sacred.observers import MongoObserver

import scripts.helper.models as models

experiment_name = "random_forest"

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
    rf_params = {
        'n_estimators': [10, 50, 100, 200, 400, 600],  # Number of trees in the forest
        'max_features': [0.1, 0.3, 'sqrt', 'log2'],  # Methods to choose number of features
        'max_depth': [2, 5, 10, 20, 30, 50, 60]  # Maximum depth of trees
    }
    num_param_sets = 10
    num_folds = 10


@ex.capture
def get_data(dataset, representation):
    """Get data corresponding to representation."""
    return models.load_data(dataset, representation)


@ex.capture
def train_model_random_search(data, rf_params, num_folds, num_param_sets):
    """Train a series of Random Forests, testing out random sets of parameters
    from rf_params. Will use cross-validation with num_folds.

    Prints out summary information from the search and returns the estimator
    which had best cross-validation score."""
    search = models.random_search_random_forest(data=data,
                                                param_dist=rf_params,
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

    data = get_data()   # parameters injected automatically

    _run.log_scalar("X_train_size", len(data['X_train']))
    _run.log_scalar("X_val_size", len(data['X_val']))
    model = train_model_random_search(data)
    short_metrics, long_metrics, plots = models.evaluate_model(model,
                                                               data['X_val'],
                                                               data['y_val'],
                                                               save_dir)

    # Log the single number metrics using sacred
    for key, value in short_metrics.items():
        _run.log_scalar(key, value)

    # Combine the two metrics dictionaries into one and save as a json file
    full_metrics = short_metrics.copy()
    full_metrics.update(long_metrics)

    metrics_filename = os.path.join(save_dir, "metrics.json")
    with open(metrics_filename, 'w') as f:
        json.dump(full_metrics, f, indent=4)

    ex.add_artifact(metrics_filename)
    for plot_file in plots.values():
        ex.add_artifact(plot_file)

    return short_metrics['accuracy']
