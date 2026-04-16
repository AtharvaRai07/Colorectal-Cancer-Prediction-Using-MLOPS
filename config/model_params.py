from scipy.stats import uniform, randint

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': randint(100, 200),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 5)
}

RANDOM_SEARCH_PARAMS = {
    "n_iter": 5,
    "cv": 5,
    "n_jobs": -1,
    "verbose": 2,
    "scoring": 'accuracy',
    "random_state": 42
}
