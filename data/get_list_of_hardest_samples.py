import os
import csv
from typing import List
import yaml

def get_label_ID_map(num_classes):
    if num_classes == 4:
        LABEL2ID = {'Normal': 0, 'PCR': 1, 'MPR': 2, 'Cancer': 3}
        ID2LABEL = ['Normal', 'PCR', 'MPR', 'Cancer']
    elif num_classes == 3:
        LABEL2ID = {'Normal': 0, 'PCR': 1, 'MPR': 2, 'Cancer': 2}
        ID2LABEL = ['Normal', 'PCR', 'Cancer']
    elif num_classes == 3.1: # Normal and PCR are combined TODO: fix the code for load image list and load_data in Dataset
        LABEL2ID = {'Normal': 0, 'PCR': 0, 'MPR': 1, 'Cancer': 2}
        ID2LABEL = ['PCR', 'MPR', 'Cancer']
    elif num_classes == 2: # Normal is excluded, only PCR, Cancer (MPR is combined with Cancer)
        LABEL2ID = {'Normal': -1, 'PCR': 0, 'MPR': 1, 'Cancer': 1}
        ID2LABEL = ['PCR', 'Cancer']
    elif num_classes == 2.1: # Normal and PCR are combined
        LABEL2ID = {'Normal': 0, 'PCR': 0, 'MPR': 1, 'Cancer': 1}
        ID2LABEL = ['PCR', 'Cancer']
    else:
        raise ValueError('num_classes should be 2, 3 or 4 instead of {}'.format(num_classes))
    return LABEL2ID, ID2LABEL

def filter_wrong_prediction(result, ID2LABEL):
    wrong_predictions = {}
    for i in range(len(ID2LABEL)):
        wrong_predictions[ID2LABEL[i]] = []
    for prediction in result:
        pred = prediction[1]
        gt = prediction[2]
        if pred != gt:
            wrong_predictions[ID2LABEL[int(gt)]].append([prediction[0], int(pred)])
    return wrong_predictions

def get_hardest_samples(exp_folder_path, threshold=None):
    if os.path.exists(os.path.join(exp_folder_path, 'args.yaml')):
        raise ValueError('expect a folder containing multiple experiments')
    exp_folder_list = [os.path.join(exp_folder_path, exp_name) for exp_name in os.listdir(exp_folder_path)] if not isinstance(exp_folder_path, List) else exp_folder_path

    sample_wrong_count = {}

    fold_result_path_template = 'fold{}_test_result.csv'
    for exp_folder in exp_folder_list:
        args_path = os.path.join(exp_folder, 'args.yaml')
        with open(args_path, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        LABEL2ID, ID2LABEL = get_label_ID_map(args['num_classes'])

        for fold in range(1, 5+1):
            fold_result_path = os.path.join(exp_folder, fold_result_path_template.format(fold))
            if not os.path.exists(fold_result_path):
                continue
            with open(fold_result_path, 'r') as f:
                reader = csv.reader(f)
                result = list(reader)[1:]
            wrong_predictions = filter_wrong_prediction(result, ID2LABEL)

            for label in wrong_predictions:
                if label not in sample_wrong_count:
                    sample_wrong_count[label] = {}
                for pred in wrong_predictions[label]:
                    img_name = pred[0]
                    if img_name not in sample_wrong_count[label]:
                        sample_wrong_count[label][img_name] = []
                    sample_wrong_count[label][img_name].append(ID2LABEL[pred[1]])

    # sort by the number of wrong predictions
    sample_wrong_count = {label: {k: v for k, v in sorted(sample_wrong_count[label].items(), key=lambda item: len(item[1]), reverse=True)} for label in sample_wrong_count}

    threshold = len(exp_folder_list) / 1.5 if threshold is None else threshold
    if threshold > len(exp_folder_list):
        threshold = len(exp_folder_list) - 1
    print('threshold:', threshold, 'len(exp_folder_list):', len(exp_folder_list))

    hardest_samples = {}
    for label in sample_wrong_count:
        hardest_samples[label] = []
        for img_name in sample_wrong_count[label]:
            if len(sample_wrong_count[label][img_name]) >= threshold:
                hardest_samples[label].append([img_name, sample_wrong_count[label][img_name]])

    hardest_samples_img_path_list = []
    for label in hardest_samples:
        for pred_sample in hardest_samples[label]:
            hardest_samples_img_path_list.append(pred_sample[0])

    return hardest_samples, hardest_samples_img_path_list