"""
Main Code to reproduce the results in the paper
'Non-Standard Echo State Networks for Video Door State Monitoring'.
"""

# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3-Clause

import logging
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     GridSearchCV, PredefinedSplit)
from sklearn.metrics import make_scorer
from scipy.stats import uniform
from joblib import dump, load
from sklearn.base import clone
from sklearn.utils.fixes import loguniform

import numpy as np
from pyrcn.echo_state_network import ESNClassifier
from pyrcn.model_selection import SequentialSearchCV
from pyrcn.metrics import accuracy_score, mean_squared_error

from dataset import VideoDoorStateRecognitionDataset
from input_to_node import ClusterInputToNode, SimpleInputToNode
from node_to_node import SCRNodeToNode, DLRBNodeToNode, DLRNodeToNode


LOGGER = logging.getLogger(__name__)


def main(fit_basic_esn=False, fit_kmeans_esn=False, fit_scr_esn=False,
         fit_dlr_esn=False, fit_dlrb_esn=False):
    """
    This is the main function to reproduce all visualizations and models for
    the paper "Non-Standard Echo State Networks for Video Door State
    Monitoring".

    It is controlled via command line arguments:

    Params
    ------
    fit_basic_esn : bool, default=False
        Fit the basic ESN models.
    fit_kmeans_esn : bool, default=False
        Fit the KM-ESN models.
    fit_scr_esn : bool, default=False
        Fit the Simple Cycle ESN models.
    fit_dlr_esn : bool, default=False
        Fit the Delay Line ESN models.
    fit_dlrb_esn : bool, default=False
        Fit the Delay Line with Feedback ESN models.
    """

    LOGGER.info("Loading the training dataset...")
    dataset = VideoDoorStateRecognitionDataset(path=r"./data")

    X, y = dataset.return_X_y()
    LOGGER.info("... done!")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    X = np.hstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    test_fold = np.zeros(shape=(len(X), ))
    test_fold[:len(X_train)] = -1
    ps = PredefinedSplit(test_fold=test_fold)
    LOGGER.info("Scaling the dataset between zero and one...")
    scaler = MinMaxScaler().fit(np.vstack(X_train))
    ohe = OneHotEncoder(sparse_output=False).fit(np.hstack(y).reshape(-1, 1))
    y_train_ohe = np.empty_like(y_train)
    y_test_ohe = np.empty_like(y_test)
    for k in range(len(X_train)):
        X_train[k] = scaler.transform(X_train[k])
        y_train_ohe[k] = ohe.transform(y_train[k].reshape(-1, 1))
    for k in range(len(X_test)):
        X_test[k] = scaler.transform(X_test[k])
        y_test_ohe[k] = ohe.transform(y_test[k].reshape(-1, 1))

    pca = PCA(n_components=30).fit(np.vstack(X_train))
    for k in range(len(X_train)):
        X_train[k] = pca.transform(X_train[k])
    for k in range(len(X_test)):
        X_test[k] = pca.transform(X_test[k])
    LOGGER.info("... done!")

    hls_search_params = {'alpha': loguniform(1e-5, 10 - 1e-5)}
    hls_search_kwargs = {
        'n_iter': 50, 'random_state': 42, 'verbose': 1, 'n_jobs': 8, "cv": 3,
        "scoring": make_scorer(mean_squared_error, greater_is_better=False,
                               needs_proba=True)}
    final_params = {"random_state": np.arange(10)}
    final_kwargs = {"scoring": make_scorer(accuracy_score), "n_jobs": -1,
                    "refit": False, "cv": ps, "verbose": 10}
    hidden_layer_sizes = (50, 100, 200, 400, 800, 1600, 3200, )

    if fit_basic_esn:
        LOGGER.info("Fitting basic ESN models ...")
        initial_esn_params = {
            'hidden_layer_size': 500, 'k_in': 10, 'input_scaling': 0.4,
            'input_activation': 'identity', 'bias_scaling': 0.0,
            'spectral_radius': 0.0, 'leakage': 1.0, 'k_rec': 10,
            'reservoir_activation': 'tanh', 'bidirectional': False,
            'alpha': 1e-5, 'random_state': 42}

        base_esn = ESNClassifier(**initial_esn_params)
        # Run model selection
        LOGGER.info(f"Performing the optimization...")
        step1_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                        'spectral_radius': uniform(loc=0, scale=2)}
        step2_params = {'leakage': uniform(1e-2, 1e0 - 1e-2)}
        step3_params = {'bias_scaling': uniform(loc=0, scale=2)}

        kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}

        searches = [('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                    ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                    ('step3', RandomizedSearchCV, step3_params, kwargs_step3)]

        try:
            search = load(f'./results/sequential_search_basic_esn.joblib')
        except FileNotFoundError:
            search = SequentialSearchCV(base_esn, searches=searches).fit(
                X_train, y_train_ohe)
            dump(search, f'./results/sequential_search_basic_esn.joblib')

        for hidden_layer_size in hidden_layer_sizes:
            try:
                load(f'./results/basic_esn_hls_{hidden_layer_size}.joblib')
            except FileNotFoundError:
                base_esn = clone(search.best_estimator_).set_params(
                    hidden_layer_size=hidden_layer_size)
                esn = RandomizedSearchCV(
                    estimator=base_esn, param_distributions=hls_search_params,
                    **hls_search_kwargs).fit(X_train, y_train_ohe)
                dump(
                    esn, f'./results/basic_esn_hls_{hidden_layer_size}.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            esn = load(f'./results/basic_esn_hls_{hidden_layer_size}.joblib')
            final_results = GridSearchCV(estimator=clone(esn.best_estimator_),
                                         param_grid=final_params,
                                         **final_kwargs).fit(X, y)
            dump(final_results, f'./results/basic_esn_hls_{hidden_layer_size}'
                                f'_final_results.joblib')
        LOGGER.info("... done!")

    if fit_kmeans_esn:
        LOGGER.info("Fitting KM-ESN models ...")
        initial_esn_params = {
            'hidden_layer_size': 500, 'k_in': 10, 'input_scaling': 0.4,
            'input_activation': 'identity', 'bias_scaling': 0.1,
            'spectral_radius': 0.0, 'leakage': 1.0, 'k_rec': 10,
            'reservoir_activation': 'tanh', 'bidirectional': False,
            'alpha': 1e-5, 'random_state': 42,
            'cluster_algorithm': 'minibatch_kmeans'}

        base_esn = ESNClassifier(input_to_node=ClusterInputToNode(),
                                 **initial_esn_params)
        # Run model selection
        LOGGER.info(f"Performing the optimization...")
        step1_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                        'spectral_radius': uniform(loc=0, scale=2)}
        step2_params = {'leakage': uniform(1e-2, 1e0 - 1e-2)}
        step3_params = {'bias_scaling': uniform(loc=0, scale=2)}

        kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}

        searches = [('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                    ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                    ('step3', RandomizedSearchCV, step3_params, kwargs_step3)]
        try:
            search = load(f'./results/sequential_search_kmeans_esn.joblib')
        except FileNotFoundError:
            search = SequentialSearchCV(base_esn, searches=searches).fit(
                X_train, y_train_ohe)
            dump(search, f'./results/sequential_search_kmeans_esn.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            try:
                load(f'./results/kmeans_esn_hls_{hidden_layer_size}.joblib')
            except FileNotFoundError:
                base_esn = clone(search.best_estimator_).set_params(
                    hidden_layer_size=hidden_layer_size)
                esn = RandomizedSearchCV(
                    estimator=base_esn, param_distributions=hls_search_params,
                    **hls_search_kwargs).fit(X_train, y_train_ohe)
                dump(esn,
                     f'./results/kmeans_esn_hls_{hidden_layer_size}.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            esn = load(f'./results/kmeans_esn_hls_{hidden_layer_size}.joblib')
            final_results = GridSearchCV(estimator=clone(esn.best_estimator_),
                                         param_grid=final_params,
                                         **final_kwargs).fit(X, y)
            dump(final_results, f'./results/kmeans_esn_hls_{hidden_layer_size}'
                                f'_final_results.joblib')
        LOGGER.info("... done!")

    if fit_scr_esn:
        LOGGER.info("Fitting Simple Cycle Reservoir ESN models ...")
        initial_esn_params = {
            'hidden_layer_size': 500, 'k_in': 10, 'input_scaling': 0.4,
            'input_activation': 'identity', 'bias_scaling': 0.0,
            'forward_weight': 0.0, 'leakage': 1.0, 'k_rec': 10,
            'reservoir_activation': 'tanh', 'bidirectional': False,
            'alpha': 1e-5, 'random_state': 42}

        base_esn = ESNClassifier(input_to_node=SimpleInputToNode(),
                                 node_to_node=SCRNodeToNode(),
                                 **initial_esn_params)
        step1_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                        'forward_weight': uniform(loc=0, scale=2)}
        step2_params = {'leakage': uniform(1e-2, 1e0 - 1e-2)}
        step3_params = {'bias_scaling': uniform(loc=0, scale=2)}

        kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}

        searches = [('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                    ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                    ('step3', RandomizedSearchCV, step3_params, kwargs_step3)]
        LOGGER.info("Fitting SCR ESN models ...")
        try:
            search = load(f'./results/sequential_search_scr_esn.joblib')
        except FileNotFoundError:
            search = SequentialSearchCV(base_esn, searches=searches).fit(
                X_train, y_train_ohe)
            dump(search, f'./results/sequential_search_scr_esn.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            try:
                load(f'./results/scr_esn_hls_{hidden_layer_size}.joblib')
            except FileNotFoundError:
                base_esn = clone(search.best_estimator_).set_params(
                    hidden_layer_size=hidden_layer_size)
                esn = RandomizedSearchCV(
                    estimator=base_esn, param_distributions=final_params,
                    **final_kwargs).fit(X_train, y_train_ohe)
                dump(esn, f'./results/scr_esn_hls_{hidden_layer_size}.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            esn = load(f'./results/scr_esn_hls_{hidden_layer_size}.joblib')
            final_results = GridSearchCV(estimator=clone(esn.best_estimator_),
                                         param_grid=final_params,
                                         **final_kwargs).fit(X, y)
            dump(final_results, f'./results/scr_esn_hls_{hidden_layer_size}'
                                f'_final_results.joblib')
        LOGGER.info("... done!")

    if fit_dlr_esn:
        LOGGER.info("Fitting Delay Line Reservoir ESN models ...")
        initial_esn_params = {
            'hidden_layer_size': 500, 'k_in': 10, 'input_scaling': 0.4,
            'input_activation': 'identity', 'bias_scaling': 0.0,
            'forward_weight': 0.0, 'leakage': 1.0, 'k_rec': 10,
            'reservoir_activation': 'tanh', 'bidirectional': False,
            'alpha': 1e-5, 'random_state': 42}

        base_esn = ESNClassifier(input_to_node=SimpleInputToNode(),
                                 node_to_node=DLRNodeToNode(),
                                 **initial_esn_params)
        step1_params = {'input_scaling': uniform(loc=1e-2, scale=1),
                        'forward_weight': uniform(loc=0, scale=2)}
        step2_params = {'leakage': uniform(1e-2, 1e0 - 1e-2)}
        step3_params = {'bias_scaling': uniform(loc=0, scale=2)}

        kwargs_step1 = {'n_iter': 200, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step2 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}

        searches = [('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                    ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                    ('step3', RandomizedSearchCV, step3_params, kwargs_step3)]
        LOGGER.info("Fitting DLR ESN models ...")
        try:
            search = load(f'./results/sequential_search_dlr_esn.joblib')
        except FileNotFoundError:
            search = SequentialSearchCV(base_esn, searches=searches).fit(
                X_train, y_train_ohe)
            dump(search, f'./results/sequential_search_dlr_esn.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            try:
                load(f'./results/dlr_esn_hls_{hidden_layer_size}.joblib')
            except FileNotFoundError:
                base_esn = clone(search.best_estimator_).set_params(
                    hidden_layer_size=hidden_layer_size)
                esn = RandomizedSearchCV(
                    estimator=base_esn, param_distributions=final_params,
                    **final_kwargs).fit(X_train, y_train_ohe)
                dump(esn, f'./results/dlr_esn_hls_{hidden_layer_size}.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            esn = load(f'./results/dlr_esn_hls_{hidden_layer_size}.joblib')
            final_results = GridSearchCV(estimator=clone(esn.best_estimator_),
                                         param_grid=final_params,
                                         **final_kwargs).fit(X, y)
            dump(final_results, f'./results/dlr_esn_hls_{hidden_layer_size}'
                                f'_final_results.joblib')
        LOGGER.info("... done!")

    if fit_dlrb_esn:
        LOGGER.info(
            "Fitting Delay Line with Feedback Reservoir ESN models ...")
        initial_esn_params = {
            'hidden_layer_size': 500, 'k_in': 10, 'input_scaling': 0.4,
            'input_activation': 'identity', 'bias_scaling': 0.0,
            'forward_weight': 0.0, 'feedback_weight': 0.0, 'leakage': 1.0,
            'k_rec': 10, 'reservoir_activation': 'tanh',
            'bidirectional':  False, 'alpha': 1e-5, 'random_state': 42}

        base_esn = ESNClassifier(input_to_node=SimpleInputToNode(),
                                 node_to_node=DLRBNodeToNode(),
                                 **initial_esn_params)
        step1_params = {'input_scaling': uniform(loc=1e-2, scale=1)}
        step2_params = {'forward_weight': uniform(loc=0, scale=2),
                        'feedback_weight': uniform(loc=0, scale=2)}
        step3_params = {'leakage': uniform(1e-2, 1e0 - 1e-2)}
        step4_params = {'bias_scaling': uniform(loc=0, scale=2)}

        kwargs_step1 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step2 = {'n_iter': 200, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step3 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}
        kwargs_step4 = {'n_iter': 50, 'random_state': 42, 'verbose': 10,
                        'n_jobs': -1, "cv": 3,
                        "scoring": make_scorer(mean_squared_error,
                                               greater_is_better=False,
                                               needs_proba=True)}

        searches = [('step1', RandomizedSearchCV, step1_params, kwargs_step1),
                    ('step2', RandomizedSearchCV, step2_params, kwargs_step2),
                    ('step3', RandomizedSearchCV, step3_params, kwargs_step3),
                    ('step4', RandomizedSearchCV, step4_params, kwargs_step4)]
        LOGGER.info("Fitting DLRB ESN models ...")
        try:
            search = load(f'./results/sequential_search_dlrb_esn.joblib')
        except FileNotFoundError:
            search = SequentialSearchCV(base_esn, searches=searches).fit(
                X_train, y_train_ohe)
            dump(search, f'./results/sequential_search_dlrb_esn.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            try:
                load(f'./results/dlrb_esn_hls_{hidden_layer_size}.joblib')
            except FileNotFoundError:
                base_esn = clone(search.best_estimator_).set_params(
                    hidden_layer_size=hidden_layer_size)
                esn = RandomizedSearchCV(
                    estimator=base_esn, param_distributions=final_params,
                    **final_kwargs).fit(X_train, y_train_ohe)
                dump(esn, f'./results/dlrb_esn_hls_{hidden_layer_size}.joblib')
        for hidden_layer_size in hidden_layer_sizes:
            esn = load(f'./results/dlrb_esn_hls_{hidden_layer_size}.joblib')
            final_results = GridSearchCV(estimator=clone(esn.best_estimator_),
                                         param_grid=final_params,
                                         **final_kwargs).fit(X, y)
            dump(final_results, f'./results/dlrb_esn_hls_{hidden_layer_size}'
                                f'_final_results.joblib')
        LOGGER.info("... done!")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_basic_esn", action="store_true")
    parser.add_argument("--fit_kmeans_esn", action="store_true")
    parser.add_argument("--fit_scr_esn", action="store_true")
    parser.add_argument("--fit_dlr_esn", action="store_true")
    parser.add_argument("--fit_dlrb_esn", action="store_true")
    args = vars(parser.parse_args())
    logging.basicConfig(format="%(asctime)s - [%(levelname)8s]: %(message)s",
                        handlers=[
                            logging.FileHandler("main.log", encoding="utf-8"),
                            logging.StreamHandler()
                        ])
    LOGGER.setLevel(logging.DEBUG)
    main(**args)
    exit(0)
