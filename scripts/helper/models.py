"""Trains models using processed datasets, and evaluates their performance."""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import sklearn.metrics as metrics


MODELS_DIR = "models/"


def create_experiment_save_dir(name):
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


def load_data(representation):
    """Loads data corresponding to a particular type of representation."""
    X_train = np.load(f"datasets/alpha/training/data_{representation}.npy")
    y_train = np.load("datasets/alpha/training/labels.npy")

    X_val = np.load(f"datasets/alpha/validation/data_{representation}.npy")
    y_val = np.load("datasets/alpha/validation/labels.npy")

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
                                cv=num_folds,
                                n_iter=num_param_sets)
    search.fit(data['X_train'], data['y_train'])
    return search


def summarise_search(search, num_results=10, full_print=False):
    """Print the results from the top num_results estimators. If full_print, then
    also print out all the results from the search."""
    results = search.cv_results_
    ranked_indices = results['rank_test_score'] - 1
    total_runs = len(ranked_indices)

    print(f"Total runs: {total_runs}")

    if full_print:
        print(results)

    for i in range(min(num_results, total_runs)):
        index = ranked_indices[i]
        print(f"Ranked {i}")
        print(f"Time to fit: {results['mean_fit_time'][index]:.2f}s")
        print(f"CV accuracy score: {results['mean_test_score'][index]:.4f}")
        print(f"Parameters: {results['params'][index]}")


def grid_search_random_forest(data, param_grid, num_folds=10):
    """Performs a grid search to find the optimal hyperparameters for a random forest
    using the training data in the data object. Uses cross-validation on all the training
    data."""
    rf_model = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=num_folds)
    grid_search.fit(data['X_train'], data['y_train'])
    return grid_search


def evaluate_model(model, X_true, y_true, savedir):
    """Uses the model to predict for the given data, and evaluates the model
    according to a number of metrics, returning these in a dictionary."""
    y_pred = model.predict(X_true)
    y_probs = model.predict_proba(X_true)[:, 1]

    precision, recall, _thresholds_pr = metrics.precision_recall_curve(y_true, y_probs)
    fpr, tpr, _thresholds_roc = metrics.roc_curve(y_true, y_probs)
    model_metrics = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
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
    }

    plot_filenames = {
        'pred_probs': os.path.join(savedir, "pred_probs.png"),
        'roc_curve': os.path.join(savedir, "roc_curve.png"),
        'pr_curve': os.path.join(savedir, "pr_curve.png")
    }

    plt.clf()
    sns.distplot(y_probs[y_true == 1], label="Positives", color=sns.color_palette('colorblind')[2])
    sns.distplot(y_probs[y_true == 0], label="Negatives", color=sns.color_palette('colorblind')[3])
    plt.title("Prediction probabilities by class")
    plt.legend()
    plt.savefig(plot_filenames['pred_probs'])

    plt.clf()
    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig(plot_filenames['roc_curve'])

    plt.clf()
    plt.plot(recall, precision)
    plt.title("Precision recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(plot_filenames['pr_curve'])

    return model_metrics, longer_model_metrics, plot_filenames