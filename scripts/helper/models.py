"""Trains models using processed datasets, and evaluates their performance."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics


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
    rf_model = RandomForestClassifier(**kwargs)
    rf_model.fit(data['X_train'], data['y_train'])
    return rf_model


def grid_search_random_forest(data):
    """Performs a grid search to find the optimal hyperparameters for a random forest
    using the training data in the data object. Uses cross-validation on all the training
    data."""
    rf_params = {
        'n_estimators': [100, 200],            # Number of trees in the forest
        'max_features': [None, 'sqrt', 'log2'],   # Methods to choose number of features
        'max_depth':    [2, 30, 60]       # Maximum depth of trees
    }

    rf_model = RandomForestClassifier(random_state=115)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_params, cv=10)
    grid_search.fit(data['X_train'], data['y_train'])
    return grid_search


def evaluate_model(model, X_true, y_true):
    """Uses the model to predict for the given data, and evaluates the model
    according to a number of metrics, returning these in a dictionary."""
    y_pred = model.predict(X_true)
    y_probs = model.predict_proba(X_true)[:, 1]

    precision, recall, thresholds_pr = metrics.precision_recall_curve(y_true, y_probs)
    fpr, tpr, thresholds_roc = metrics.roc_curve(y_true, y_probs)
    model_metrics = {
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'f1_score': metrics.f1_score(y_true, y_pred),
        'precision': metrics.precision_score(y_true, y_pred),
        'recall': metrics.recall_score(y_true, y_pred),
        'cross_entropy': metrics.log_loss(y_true, y_pred),
        'average_precision_score': metrics.average_precision_score(y_true, y_pred),
        'confusion_matrix': metrics.confusion_matrix(y_true, y_pred),

        'binding_probs': stats.describe(y_probs),
        'binding_probs_positive': stats.describe(y_probs[y_true == 1]),
        'binding_probs_negative': stats.describe(y_probs[y_true == 0]),
        'pr_auc_score': metrics.auc(recall, precision),
        'roc_auc_score': metrics.auc(fpr, tpr),
        'brier_score_loss': metrics.brier_score_loss(y_true, y_probs),
    }

    plt.clf()
    sns.distplot(y_probs[y_true == 1], label="Positives", color=sns.color_palette('colorblind')[2])
    sns.distplot(y_probs[y_true == 0], label="Negatives", color=sns.color_palette('colorblind')[3])
    plt.title("Prediction probabilities by class")
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(recall, precision)
    plt.title("Precision recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    return model_metrics


def run(representation):
    """Runs model training and evaluation for the given type of data representation."""
    data = load_data(representation)
    grid_search = grid_search_random_forest(data)
    classifier = grid_search.best_estimator_
    model_metrics = evaluate_model(classifier, data['X_val'], data['y_val'])
