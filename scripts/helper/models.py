"""Trains models using processed datasets, and evaluates their performance."""
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import scipy.sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics


MODELS_DIR = "models/"


# pylint: disable-msg=arguments-differ,method-hidden
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder that can deal with np values."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_to_json(object_to_save, filename):
    """Save object to file, using a NumpyEncoder."""
    with open(filename, 'w') as f:
        json.dump(object_to_save, f, cls=NumpyEncoder, indent=4)


def create_experiment_save_dir_unique(name):
    """Finds a unique folder to save artifacts for a given experiment."""
    filename = os.path.join(MODELS_DIR, name)
    i = 1
    unique_dir = "{}-{}".format(filename, i)
    while os.path.exists(unique_dir):
        i += 1
        unique_dir = "{}-{}".format(filename, i)

    try:
        os.makedirs(unique_dir)
    except FileExistsError:
        # directory already exists
        pass
    return unique_dir


def create_experiment_save_dir(dataset, representation, experiment_name):
    save_dir = os.path.join(MODELS_DIR, dataset, representation, experiment_name)
    os.makedirs(save_dir)
    return save_dir


def load_data(dataset, representation):
    """Loads data corresponding to a particular type of representation."""
    logging.info(f"Loading data from dataset {dataset}, representation {representation}")

    if representation == "fingerprints":
        X_train = scipy.sparse.load_npz(f"datasets/{dataset}/training/data_{representation}.npz")
        y_train = np.load(f"datasets/{dataset}/training/labels.npy")

        logging.info(f"Loaded training data")

        X_val = scipy.sparse.load_npz(f"datasets/{dataset}/validation/data_{representation}.npz")
        y_val = np.load(f"datasets/{dataset}/validation/labels.npy")

        logging.info(f"Loaded validation data")
    else:
        X_train = np.load(f"datasets/{dataset}/training/data_{representation}.npy")
        y_train = np.load(f"datasets/{dataset}/training/labels.npy")

        logging.info(f"Loaded training data")

        X_val = np.load(f"datasets/{dataset}/validation/data_{representation}.npy")
        y_val = np.load(f"datasets/{dataset}/validation/labels.npy")

        logging.info(f"Loaded validation data")

    data = {
        'representation': representation,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
    }
    return data


def train_random_forest(data, **kwargs):
    """Trains a random forest on the training data in the data object, passing any
    extra arguments to the random forest classifier."""
    # The default value is going to update soon to 100, so we might as well use
    #   this as our default.
    if 'n_estimators' not in kwargs:
        kwargs['n_estimators'] = 100

    rf_model = RandomForestClassifier(**kwargs)
    rf_model.fit(data['X_train'], data['y_train'])
    return rf_model


def random_search_random_forest(data, param_dist, num_folds=10, num_param_sets=10):
    """Performs a grid search to find the optimal hyperparameters for a random forest
    using the training data in the data object. Uses cross-validation on all the training
    data."""
    rf_model = RandomForestClassifier()
    search = RandomizedSearchCV(estimator=rf_model,
                                param_distributions=param_dist,
                                n_jobs=1,
                                cv=num_folds,
                                return_train_score=True,
                                n_iter=num_param_sets,
                                verbose=10)
    search.fit(data['X_train'], data['y_train'])
    return search


def random_search_logistic_regression(data, param_dist, num_folds=10, num_param_sets=10):
    """Performs a grid search to find the optimal hyperparameters for a logistic regression
    using the training data in the data object. Uses cross-validation on all the training
    data."""
    clf = SGDClassifier(penalty='elasticnet',
                        loss='log',
                        n_jobs=1)
    search = RandomizedSearchCV(estimator=clf,
                                param_distributions=param_dist,
                                cv=num_folds,
                                return_train_score=True,
                                n_iter=num_param_sets,
                                verbose=10)
    search.fit(data['X_train'], data['y_train'])
    return search


def summarise_search(search, num_results=10, full_print=False):
    """Print the results from the top num_results estimators. If full_print, then
    also print out all the results from the search."""
    results = search.cv_results_
    ranks = results['rank_test_score']
    sorted_indices = sorted(zip(ranks, range(len(ranks))))
    total_runs = len(ranks)

    logging.info(f"Total runs: {total_runs}")

    if full_print:
        logging.info(results)

    for rank, index in sorted_indices[:num_results]:
        logging.info(f"Ranked {rank}")
        logging.info(f"Time to fit: {results['mean_fit_time'][index]:.2f}s")
        logging.info(f"CV accuracy score: {results['mean_test_score'][index]:.4f}")
        logging.info(f"Parameters: {results['params'][index]}")


def grid_search_random_forest(data, param_grid, num_folds=10):
    """Performs a grid search to find the optimal hyperparameters for a random forest
    using the training data in the data object. Uses cross-validation on all the training
    data."""
    rf_model = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=num_folds)
    grid_search.fit(data['X_train'], data['y_train'])
    return grid_search


# pylint: disable-msg=too-many-locals
def evaluate_predictions(y_pred, y_probs, y_true, y_train_pred, y_train_probs, y_train, savedir):
    logging.info(f"Calculating accuracy metrics")
    precision, recall, _thresholds_pr = metrics.precision_recall_curve(y_true, y_probs)
    fpr, tpr, _thresholds_roc = metrics.roc_curve(y_true, y_probs)
    model_metrics = {
        'training_accuracy': metrics.accuracy_score(y_train, y_train_pred),
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'training_f1_score': metrics.f1_score(y_train, y_train_pred),
        'f1_score': metrics.f1_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred),
        'recall': metrics.recall_score(y_true, y_pred),
        'cross_entropy': metrics.log_loss(y_true, y_pred),
        'average_precision_score': metrics.average_precision_score(y_true, y_pred),
        'pr_auc_score': metrics.auc(recall, precision),
        'roc_auc_score': metrics.auc(fpr, tpr),
        'brier_score_loss': metrics.brier_score_loss(y_true, y_probs),
    }

    longer_model_metrics = {
        'confusion_matrix': metrics.confusion_matrix(y_true, y_pred).tolist(),
        'binding_probs': stats.describe(y_probs)._asdict(),
        'binding_probs_positive': stats.describe(y_probs[y_true == 1])._asdict(),
        'binding_probs_negative': stats.describe(y_probs[y_true == 0])._asdict(),
        'training_binding_probs': stats.describe(y_train_probs)._asdict(),
        'training_binding_probs_positive': stats.describe(y_train_probs[y_train == 1])._asdict(),
        'training_binding_probs_negative': stats.describe(y_train_probs[y_train == 0])._asdict(),
    }

    plot_filenames = {
        'pred_probs': os.path.join(savedir, "pred_probs.png"),
        'roc_curve': os.path.join(savedir, "roc_curve.png"),
        'pr_curve': os.path.join(savedir, "pr_curve.png")
    }

    logging.info(f"Plotting predicted probability distribution")

    try:
        plt.clf()
        sns.distplot(y_probs[y_true == 1],
                     label="Positives",
                     color=sns.color_palette('colorblind')[2])
        sns.distplot(y_probs[y_true == 0],
                     label="Negatives",
                     color=sns.color_palette('colorblind')[3])
    except np.linalg.LinAlgError:
        # If all the predicted probabilities are the same, then we cannot calculate kde
        plt.clf()
        sns.distplot(y_probs[y_true == 1],
                     label="Positives",
                     kde=False,
                     color=sns.color_palette('colorblind')[2])
        sns.distplot(y_probs[y_true == 0],
                     label="Negatives",
                     kde=False,
                     color=sns.color_palette('colorblind')[3])
    plt.title("Prediction probabilities by class")
    plt.legend()
    plt.savefig(plot_filenames['pred_probs'])

    logging.info(f"Plotting ROC curve")
    plt.clf()
    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig(plot_filenames['roc_curve'])

    logging.info(f"Plotting precision recall curve")
    plt.clf()
    plt.plot(recall, precision)
    plt.title("Precision recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(plot_filenames['pr_curve'])

    return model_metrics, longer_model_metrics, plot_filenames


def evaluate_model(model, data, savedir):
    """Uses the model to predict for the given data, and evaluates the model
    according to a number of metrics, returning these in a dictionary."""
    logging.info(f"Getting predictions for validation set and training set")
    y_pred = model.predict(data['X_val'])
    y_probs = model.predict_proba(data['X_val'])[:, 1]
    y_true = data['y_val']

    y_train_pred = model.predict(data['X_train'])
    y_train_probs = model.predict_proba(data['X_train'])[:, 1]
    y_train = data['y_train']

    return evaluate_predictions(y_pred,
                                y_probs,
                                y_true,
                                y_train_pred,
                                y_train_probs,
                                y_train,
                                savedir)
