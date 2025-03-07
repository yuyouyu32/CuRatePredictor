import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import classification_report
from config import *
from dataloader.my_dataloader import CustomDataLoader
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel, DotProduct,
                                              ExpSineSquared, Matern,
                                              RationalQuadratic)
import shap
import matplotlib.pyplot as plt


# python -m MLs.SHAP
if __name__ == "__main__":
    all_columns = pd.read_excel(RegDataPath).columns.tolist()
    drop_columns = DropColumns
    # 按照feature的列顺序给我feature_columns 
    feature_columns =FeatureColumns
    for i in range(len(feature_columns)):
        if feature_columns[i] in ProcessColumnsEn:
            feature_columns[i] = ProcessColumnsEn[feature_columns[i]]
    dataloader = CustomDataLoader(RegDataPath, drop_columns, LabelsColumns)
    features, target = dataloader.get_features_for_target('综合')
    X = features
    y = target

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # 用于存储所有折的 SHAP 值
    # reg = GaussianProcessRegressor(kernel= 1**2 * Matern(length_scale=1, nu=1.5))
    reg = SVR(C=10, gamma=0.1)
    all_shap_values = np.zeros((X.shape[0], len(feature_columns)))  # 存储 SHAP 值
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg.fit(X_train, y_train)

        explainer = shap.Explainer(reg.predict, X_train)
        shap_values = explainer(X_test)
        all_shap_values[test_index, :] = shap_values.values


    # 重新构造 SHAP 值对象
    mean_shap = shap.Explanation(
        values=all_shap_values,
        base_values=None, 
        data=features,
        feature_names=feature_columns
    )
    # 绘制 SHAP 重要性条形图并保存
    plt.figure()
    shap.summary_plot(mean_shap, show=False)
    plt.savefig(RegResultpath + f"Gaussian_shap wo R.png", dpi=300, bbox_inches="tight")
    plt.close()





