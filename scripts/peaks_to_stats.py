import os
import pandas as pd
import numpy as np

import json
from tqdm import tqdm

# ================== INPUT/OUTPUT FILES ===================

# Folder containing peak lists for each fold
PEAK_DICT_INPUTS_FOLDER = r''

# Path to splits breakdown
ALL_SPLITS = r''

# Path to raven selection, used as ground truth label (.txt file)
RAVEN_SEL_PATH = r''

STATS_OUTPUT_FOLDER = r''

# ====================== PARAMETERS =======================

SAMPLE_RATE = 48000

FILTER_STATES = ['nofilter', 'filter4kHz']
COMPUTED_FOLDS = [0, 1, 2, 3, 4]

# ======================= FUNCTIONS =======================

# Define the overlap criterion between two events from two selections
# Criteria is set to 50% Intersection Over Union (IOU)
def validOverlap(left_row, right_row):
    
    if left_row['dp_s'] <= right_row['dp_s']:
        start_1, end_1 = left_row['dp_s'], left_row['dp_e']
        start_2, end_2 = right_row['dp_s'], right_row['dp_e']
    else:
        start_2, end_2 = left_row['dp_s'], left_row['dp_e']
        start_1, end_1 = right_row['dp_s'], right_row['dp_e']

    if end_1 < start_2:
        return False

    union = max(end_1, end_2) - start_1
    intersection = (end_1 - start_1) + (end_2 - start_2) - union

    return intersection >= 0.5 * union

# Return the length difference between an event and its label, in seconds
def peakLengthDiff(pred_row, label_row):
    return ((pred_row['dp_e'] - pred_row['dp_s']) - (label_row['dp_e'] - label_row['dp_s'])) / SAMPLE_RATE

# Compute different statistics on a file-by-file basis
# 'peaks' stats are event-based, 'rate' stats are segment-based
def filewiseStats(prediction_df, label_df):

    # no predictions on either side
    if len(prediction_df) == 0 and len(label_df) == 0:
        return {
            'peaks_TP': 0, 'peaks_FP': 0, 'peaks_FN': 0, 'peak_diffs': [],
            'rate_TP': 0, 'rate_FP': 0, 'rate_FN': 0
            }

    # no labels — False Positives (Type I Error)
    if len(label_df) == 0:
        return {
            'peaks_TP': 0, 'peaks_FP': len(prediction_df), 'peaks_FN': 0, 'peak_diffs': [],
            'rate_TP': 0, 'rate_FP': 1, 'rate_FN': 0
            }

    # no predictions — False Negatives (Type II Error)
    if len(prediction_df) == 0:
        return {
            'peaks_TP': 0, 'peaks_FP': 0, 'peaks_FN': len(label_df), 'peak_diffs': [],
            'rate_TP': 0, 'rate_FP': 0, 'rate_FN': 1
            }

    # assume all detections are wrong
    false_positives = [1] * len(prediction_df)
    false_negatives = [1] * len(label_df)
    peaks_TP = 0
    peak_differences = []

    # iteratively remove iterations that match
    for pred_index, (_, pred_row) in enumerate(prediction_df.iterrows()):
        for label_index, (_, label_row) in enumerate(label_df.iterrows()):
            if validOverlap(pred_row, label_row):
                false_positives[pred_index] = 0
                false_negatives[label_index] = 0
                peaks_TP += 1
                peak_differences.append(peakLengthDiff(pred_row, label_row))

    return {
            'peaks_TP': peaks_TP, 'peaks_FP': sum(false_positives), 'peaks_FN': sum(false_negatives), 'peak_diffs': peak_differences,
            'rate_TP': 1, 'rate_FP': 0, 'rate_FN': 0
            }

# Return combined stats for a list of given filenames
def getSetStats(combined_filenames_list, set_predictions_df, set_labels_df):
    
    stats_types = ['peaks_TP', 'peaks_FP', 'peaks_FN', 'rate_TP', 'rate_FP', 'rate_FN']
    stats_dict = {x : 0 for x in stats_types}
    global_peak_differences = []

    for filename in combined_filenames_list:
        prediction_df = set_predictions_df[set_predictions_df['filename'] == filename]
        label_df = set_labels_df[set_labels_df['filename'] == filename]
        file_stats = filewiseStats(prediction_df, label_df)
        global_peak_differences += file_stats['peak_diffs']

        for stat in stats_types:
            stats_dict[stat] += file_stats[stat]

    if stats_dict['peaks_TP'] == 0:

        stats_dict['peaks_precision'] = 'N/A'
        stats_dict['peaks_recall'] = '0'
        stats_dict['peaks_F1'] = '0'
        stats_dict['avg_peak_length_error'] = 'N/A'
        stats_dict['avg_absolute_peak_length_error'] = 'N/A'

    else:

        stats_dict['peaks_precision'] = stats_dict['peaks_TP'] / (stats_dict['peaks_TP'] + stats_dict['peaks_FP'])
        stats_dict['peaks_recall'] = stats_dict['peaks_TP'] / (stats_dict['peaks_TP'] + stats_dict['peaks_FN'])
        stats_dict['peaks_F1'] = 2 * (stats_dict['peaks_TP']) / (2 * stats_dict['peaks_TP'] + stats_dict['peaks_FN'] + stats_dict['peaks_FP'])
        stats_dict['avg_peak_length_error'] = np.mean(global_peak_differences)
        stats_dict['avg_absolute_peak_length_error'] = np.mean(np.abs(global_peak_differences))

    
    if stats_dict['rate_TP'] == 0:

        stats_dict['rate_precision'] = 'N/A'
        stats_dict['rate_recall'] = '0'
        stats_dict['rate_F1'] = '0'

    else:

        stats_dict['rate_precision'] = stats_dict['rate_TP'] / (stats_dict['rate_TP'] + stats_dict['rate_FP'])
        stats_dict['rate_recall'] = stats_dict['rate_TP'] / (stats_dict['rate_TP'] + stats_dict['rate_FN'])
        stats_dict['rate_F1'] = 2 * (stats_dict['rate_TP']) / (2 * stats_dict['rate_TP'] + stats_dict['rate_FP'] + stats_dict['rate_FN'])

    return stats_dict

# ===================== MAIN SCRIPT =======================

if __name__ == '__main__':

    # Processing of raven .txt selection
    raven_selection = pd.read_csv(RAVEN_SEL_PATH, sep = '\t')

    df_peaks = raven_selection[raven_selection['View'] == 'Waveform 1']
    df_peaks = df_peaks.drop(labels = ['Selection', 'View', 'Channel', 'Low Freq (Hz)', 'High Freq (Hz)'], axis = 1)

    df_peaks['filename'] = df_peaks['Begin Path'].apply(lambda x : os.path.basename(x))
    df_peaks['dp_s_sec'] = df_peaks['File Offset (s)'] 
    df_peaks['dp_e_sec'] = (df_peaks['File Offset (s)'] + df_peaks['End Time (s)'] - df_peaks['Begin Time (s)']) 
    df_peaks['dp_s'] = df_peaks['File Offset (s)'] * SAMPLE_RATE
    df_peaks['dp_e'] = (df_peaks['File Offset (s)'] + df_peaks['End Time (s)'] - df_peaks['Begin Time (s)']) * SAMPLE_RATE

    df_peaks['dp_s'] = df_peaks['dp_s'].astype(int)
    df_peaks['dp_e'] = df_peaks['dp_e'].astype(int)

    df_all_labels = df_peaks.drop(labels = ['Begin Time (s)', 'End Time (s)', 'Begin Path', 'File Offset (s)'], axis = 1)

    with open(ALL_SPLITS) as all_splits_file:
        all_splits = json.load(all_splits_file)

    for filter_state in FILTER_STATES:

        print('Running all setting combinations for ',filter_state)

        df_peak_boundaries = pd.DataFrame()
        n_folds_filelist = []

        # Aggregate files from all folds on which statistics are computed
        for n_fold in COMPUTED_FOLDS:
            fold_list = [file_info[1] for file_info in all_splits[f'train_fold_{n_fold}']]
            n_folds_filelist += fold_list

            peak_file_input = f'peaks_all_frame_volumes_train_fold_{n_fold}_{filter_state}.csv'
            peak_path = os.path.join(PEAK_DICT_INPUTS_FOLDER, peak_file_input)
            df_peak_boundaries_n_fold = pd.read_csv(peak_path)

            df_peak_boundaries = pd.concat((df_peak_boundaries, df_peak_boundaries_n_fold))

        on_threshold_list = df_peak_boundaries['on_threshold_db'].unique()
        off_threshold_list = df_peak_boundaries['off_threshold_db'].unique()
        frame_length_list = df_peak_boundaries['frame_length_s'].unique()

        all_settings = df_peak_boundaries[['on_threshold_db', 'off_threshold_db', 'frame_length_s']].drop_duplicates()
        set_labels_df = df_all_labels[df_all_labels['filename'].apply(lambda x : x in n_folds_filelist)]
        
        all_runs = []

        # Compute performance stats for all parameter settings
        for i, setting in tqdm(all_settings.iterrows(), total = len(all_settings)):
            on, off, f_size = setting[['on_threshold_db', 'off_threshold_db', 'frame_length_s']]

            set_predictions_df = df_peak_boundaries[
                (df_peak_boundaries['on_threshold_db'] == on) &
                (df_peak_boundaries['off_threshold_db'] == off) &
                (df_peak_boundaries['frame_length_s'] == f_size)
            ]

            stats_dict = getSetStats(n_folds_filelist, set_predictions_df, set_labels_df)
            settings_dict = {'filter_state': filter_state,'frame_length_s': f_size, 'on_threshold_db': on, 'off_threshold_db': off}
            run_dict = settings_dict | stats_dict
            all_runs.append(run_dict)

        df_all_runs = pd.DataFrame.from_dict(all_runs)
        stats_output_filename = f'run_stats_{filter_state}.csv'
        stats_output_filepath = os.path.join(STATS_OUTPUT_FOLDER, stats_output_filename)
        df_all_runs.to_csv(stats_output_filepath, index=False)

        print('Saved stats at: ',stats_output_filepath)