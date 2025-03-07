# 把ipynb的运行根目录放在../src下
import os
import sys
import pandas as pd
import numpy as np
import json
from config import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scienceplots

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def draw_r2_mape(y_test, y_pred, target_name, mape, ave_r2):
    with plt.style.context(['science', 'scatter']):
        # Create a scatter plot with a color map based on MAPE values
        colors = mcolors.Normalize(vmin=np.min(mape), vmax=min(np.max(mape), 100))
        plt.scatter(y_test, y_pred, c=mape, cmap='viridis', alpha=0.7, norm=colors, label=f"${target_name} Pred$")

        # Add a color bar to indicate the range of MAPE values
        plt.colorbar(label='MAPE(\%)')
        # Plot the 45-degree line
        min_value = min(y_test.min(), y_pred.min()) - y_test.min()/10
        max_value = max(y_test.max(), y_pred.max()) + y_test.min()/10
        plt.fill_between([min_value, max_value], [min_value - y_test.min()/10, max_value - y_test.min()/10], [min_value + y_test.min()/10, max_value + y_test.min()/10], color='dodgerblue', alpha=0.2, lw=0)
        plt.plot([min_value, max_value], 
                [min_value, max_value], 
                'k--')

        # Labels and title
        # plt.legend()
        plt.xlabel(f'{target_name} True Values', fontdict={'color': 'black', 'font': 'Times New Roman'})
        plt.ylabel(f'{target_name} Predicted Values', fontdict={'color': 'black', 'font': 'Times New Roman'})
        textstr = f'$R^2 = {ave_r2:.2f}$'
        plt.text(0.6, 0.15, textstr, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', fontdict={'color': 'black', 'font': 'Times New Roman'})
        if '\\' in target_name:
            target_name = target_name.replace('\\', '')
        plt.savefig(f'../results/{target_name}_scatter_verification.pdf', dpi=1000, bbox_inches='tight')
        plt.close()


# python -m draw.draw_r2
if __name__ == '__main__':
    CN2EN = {
        '综合_with_R': 'Overall Cu Recovery',
        '综合_wo_R': 'Overall Cu Recovery w_o R',
    }
    y_pred_results_path = '../results/pred_results'
    all_results_file = os.listdir(y_pred_results_path)
    for file in all_results_file:
        target_name = file.split('.')[0]
        target_name = CN2EN.get(target_name, target_name)
        print(f"Processing {target_name}")
        data = pd.read_excel(os.path.join(y_pred_results_path, file))
        y_pred = np.array(data[f"y_pred"])
        y_test = np.array(data[f"y_test"])
        ave_r2 = r2_score(y_test, y_pred)
        y_test = np.where(y_test == 0, 1, y_test)
        mape = np.abs((y_test - np.abs(y_pred)) / (y_test)) * 100
        ave_mape = np.mean(mape)
        ave_mse = mean_squared_error(y_test, y_pred)
        print(f"{target_name} R2: {ave_r2:.2f}, MAPE: {ave_mape:.2f}, MSE: {ave_mse:.2f}")
        if target_name == 'Ε(%)':
            target_name = 'E(\%)'
        draw_r2_mape(y_test, y_pred, target_name, mape, ave_r2)
