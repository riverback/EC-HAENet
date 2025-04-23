import argparse
import csv
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
from matplotlib import pyplot as plt
import yaml

from data import get_label_ID_map


def generate_cross_validation_report(exp_folder_path):
    with open(os.path.join(exp_folder_path, 'args.yaml'), 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    model_name = args['model_name']

    # calculate cross-validation metrics
    preds = []
    targets = []
    for i in range(1, 6):
        # csv file: columns = [filename, pred, target]
        preds_path = os.path.join(exp_folder_path, f'fold{i}_test_result.csv')
        with open(preds_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                preds.append(int(row[1]))
                targets.append(int(row[2]))
    num_classes = max(targets) + 1
    LABEL2ID, ID2LABEL = get_label_ID_map(num_classes)
    
    with open(os.path.join(exp_folder_path, 'report.txt'), 'w') as f:
        f.write(f'Model: {model_name}, Class: {num_classes}, Imaging: {"-".join(args["imaging_type"])}\n')
        f.write(f'Classification Report:\n{classification_report(targets, preds, target_names=ID2LABEL, digits=4)}\n')
        f.write(f'Confusion Matrix:\n{confusion_matrix(targets, preds)}\n')
    print()
    
    label_list, prob_list = [], []
    for i in range(1, 6):
        with open(os.path.join(exp_folder_path, f'fold{i}_roc_plot_data.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                prob_list.append(float(row[0]))
                label_list.append(int(row[1]))
    
    display = RocCurveDisplay.from_predictions(
        y_true=np.array(label_list),
        y_pred=np.array(prob_list),
        color="darkorange",
        # name="{}".format(model_name.split('.')[0].split('_')[0]),
        plot_chance_level=True,
    )
    auc_score = display.roc_auc
    print(auc_score)
    plt.legend([f'{model_name.split(".")[0].split("_")[0]} (AUC = {auc_score:.2f})'])
    _ = display.ax_.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    )
    plt.savefig(f'{exp_folder_path}/roc_curve.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default='experiments')
    root_folder = parser.parse_args().root_folder
    for exp_folder in os.listdir(root_folder):
        exp_folder_path = os.path.join(root_folder, exp_folder)
        if os.path.isdir(exp_folder_path):
            if os.path.exists(os.path.join(exp_folder_path, 'fold5_test_result.csv')):
                generate_cross_validation_report(exp_folder_path)
                print(f'Generated report for {exp_folder}')