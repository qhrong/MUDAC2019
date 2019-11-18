

def Random_forest(train_X, train_y, test_y, params):
    RFR = RandomForestRegressor(max_depth=)
    model = RFR.fit(X_train, y_train)
    return model

param_grid = {
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}