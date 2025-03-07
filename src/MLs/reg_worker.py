import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
# 归一化
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from .reg_config import *
from config import logging

logger = logging.getLogger(__name__)



if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses


class ModelEvaluatorKFold:
    def __init__(self, n_splits=5):
        self.models = models
        self.params = param_grid
        self.best_models = {}
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
    
    def inverse_normal_targets(self, target_values):
        return self.target_scaler.inverse_transform(target_values.reshape(-1, 1)).flatten()

    def evaluate_models(self, features, target, norm_features=False, norm_target=False):
        results = {}
        if norm_features:
            # 纵向归一化
            features = self.feature_scaler.fit_transform(features)

        if norm_target:
            # 纵向归一化到0-1
            target = target.reshape(-1, 1)
            target = self.target_scaler.fit_transform(target)

        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name} with K-fold cross-validation...")
            param_grid = self.params[model_name]
            
            grid_search = GridSearchCV(model, param_grid, cv=self.kf, scoring='neg_mean_squared_error',verbose=0, n_jobs=12)
            logger.info("{:=^40}".format(f" {model_name} Grid Search"))
            grid_search.fit(features, target)
            # logger.info(pd.DataFrame(grid_search.cv_results_))
            
            best_model = grid_search.best_estimator_
            self.best_models[model_name] = best_model
            
            best_params = grid_search.best_params_
            model.set_params(**best_params)
            r2, rmse, mape = self.eval_model(model, features, target)
            results[model_name] = {'best_params': best_params, 'rmse': rmse, 'r2': r2, 'mape': mape}
            logger.info(f"{model_name} R2 Score: {r2}, RMSE: {rmse}, MAPE: {mape}\n")
            logger.info(f"{model_name} Best Parameters: {best_params}\n", results[model_name])
            logger.info("{:=^40}".format(f" Best {model_name} Parameters Done"))
            
        return results

    def eval_model(self, model, features, target):
        r2_scores = []
        rmse_scores = []
        mape_scores = []
        all_y_test = []
        all_y_pred = []
        for train_index, test_index in self.kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = target[train_index], target[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_test = self.inverse_normal_targets(y_test)
            y_pred = self.inverse_normal_targets(y_pred)

            r2_scores.append(r2_score(y_test, y_pred))
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            # Handling zero values in target for MAPE calculation
            y_test_without_zeros = np.where(y_test != 0, y_test, 1e-8)
            y_pred_without_zeros = np.where(y_test != 0, y_pred, 1e-8)
            mape = np.mean(np.abs((y_test_without_zeros - y_pred_without_zeros) / y_test_without_zeros)) * 100
            mape_scores.append(mape)
        mean_r2_score = np.mean(r2_scores)
        mean_rmse_score = np.mean(rmse_scores)
        mean_mape_score = np.mean(mape_scores)
        all_pred_r2 = r2_score(all_y_test, all_y_pred)
        logger.info(f"Overall R2 Score: {all_pred_r2}")
        return mean_r2_score, mean_rmse_score, mean_mape_score

def unit_test():
    features = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    target = pd.Series(np.random.randn(100))
    evaluator = ModelEvaluatorKFold(n_splits=5)
    evaluation_results = evaluator.evaluate_models(features, target)
    for model_name, result in evaluation_results.items():
        logger.info(f"Model: {model_name}")
        logger.info(f"Best Parameters: {result['best_params']}")
        logger.info(f"Best Mean Squared Error: {result['best_mse']}")
        logger.info(f"R2 Score: {result['r2']}")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {result['mape']}\n")

if __name__ == "__main__":
    unit_test()
