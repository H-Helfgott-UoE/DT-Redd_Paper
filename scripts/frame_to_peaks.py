import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# ================== INPUT/OUTPUT FILES ===================

# Paths of framewise volumes for each fold
INPUT_PATHS = [
    r'',
    r''
]

OUTPUT_FOLDER = ''

# ====================== PARAMETERS =======================

MIN_PEAK_LENGTH = 0.5
MAX_PEAK_LENGTH = 10
MIN_GAP_LENGTH = 0.5

THRESHOLDS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

OVERLAP = 0.8
SAMPLE_RATE = 48000

# ======================= FUNCTIONS =======================

# Return list of individual volume peaks from frame volumes
def getPeakBoundaries(frame_volumes, params):

    if len(frame_volumes) == 0:
        return []

    on_threshold_db, off_threshold_db = params['on_threshold_db'], params['off_threshold_db']

    # in theory, min_gap_frames could be estimated without sample rate information
    # the computation below's only goal is to mirror fileToFrames and guarantee code equivalence 
    frame_length = int(params['frame_length_s'] * params['sample_rate'])
    step_length = int(frame_length * (1 - params['overlap']))
    min_gap_frames = int(params['min_gap_s'] * params['sample_rate'] / step_length)

    bg_volume_db = np.median(frame_volumes)

    on_thresh_above = frame_volumes >= (bg_volume_db + on_threshold_db)
    off_thresh_above = frame_volumes >= (bg_volume_db + off_threshold_db)

    diff = np.diff(off_thresh_above.astype(np.int8))

    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]

    if off_thresh_above[0]:
        starts = np.r_[0, starts]
    if off_thresh_above[-1]:
        ends = np.r_[ends, len(off_thresh_above) - 1]

    # For each off-threshold peak, check if there is at least one on-threshold match
    valid_peaks = []
    for s, e in zip(starts, ends):
        if on_thresh_above[s:e+1].any():
            valid_peaks.append((s, e))

    if len(valid_peaks) == 0:
        return []

    # Merge peaks
    merged = [valid_peaks[0]]
    for s, e in valid_peaks[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= min_gap_frames:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))

    # Convert back from frame numbers to data points (from the first datapoint of the starting 
    # frame, to the last datapoint of the ending frame) and eliminate short peaks
    datapoint_peaks = []
    for s, e in merged:
        dp_s = s * step_length
        dp_e = e * step_length + frame_length

        if ((dp_e - dp_s) >= (params['min_peak_length_s'] * params['sample_rate'])) and ((dp_e - dp_s) <= (params['max_peak_length_s'] * params['sample_rate'])):
            datapoint_peaks.append((dp_s, dp_e))

    return datapoint_peaks

# Plot detected peaks
def plotPeakBoundaries(datapoint_peaks, frame_volumes, params):

    bg_volume_db = np.median(frame_volumes)
    on_line_db = bg_volume_db + params['on_threshold_db']
    off_line_db = bg_volume_db + params['off_threshold_db']

    frame_length = int(params['frame_length_s'] * params['sample_rate'])
    step_length = int(frame_length * (1 - params['overlap']))

    plt.figure(figsize = (10, 5))
    plt.plot(frame_volumes)

    plt.hlines(y = [on_line_db, off_line_db], xmin = 0, xmax = len(frame_volumes), color = 'indigo', linestyle = '--', label = 'On/Off Thresholds (dB)')
    plt.hlines(y = [bg_volume_db], xmin = 0, xmax = len(frame_volumes), color = 'darkviolet', linestyle = '--', label = 'Backgroud Volume (dB)')

    for dp_s, dp_e in datapoint_peaks:
        frame_s = dp_s / step_length
        frame_e = (dp_e - frame_length) / step_length
        plt.vlines(x=(frame_s), ymin = -120, ymax = 0, color='limegreen', linestyle='--')
        plt.vlines(x=(frame_e), ymin = -120, ymax = 0, color='tomato', linestyle='--')
    plt.show()

# ===================== MAIN SCRIPT =======================

if __name__ == "__main__":

    for input_set_path in INPUT_PATHS:
        print('Processing: ', input_set_path)
        filter_state = os.path.basename(input_set_path).split('.')[0].split('_')[-1]

        with open(input_set_path, 'rb') as file:
            frame_volumes_set = pickle.load(file)

        list_peak_boundaries = []

        # Loop within a set of per-frame volume arrays (over files & frame lengths)
        for file_path, param_dict in tqdm(frame_volumes_set.items(), total = len(frame_volumes_set)):
            for frame_length_param, frame_volumes in param_dict.items():

                filename = os.path.basename(file_path)
                frame_length_s = float(frame_length_param)

                # Loop over a set of threshold parameters
                for on_index, on_threshold_db in enumerate(THRESHOLDS[1:]):
                    for off_threshold_db in THRESHOLDS[:on_index + 1]:

                        params = {
                            'on_threshold_db':  on_threshold_db,
                            'off_threshold_db':  off_threshold_db,

                            'frame_length_s': frame_length_s,
                            'overlap': OVERLAP,
                            'sample_rate': SAMPLE_RATE,

                            'min_peak_length_s': MIN_PEAK_LENGTH,
                            'max_peak_length_s': MAX_PEAK_LENGTH,
                            'min_gap_s': MIN_GAP_LENGTH
                        }

                        datapoint_peaks = getPeakBoundaries(frame_volumes, params)
                        for dp_s, dp_e in datapoint_peaks:
                            list_peak_boundaries.append([filename, filter_state, frame_length_param, on_threshold_db, off_threshold_db, dp_s, dp_e])

        df_peak_boundaries = pd.DataFrame(list_peak_boundaries, columns = ['filename', 'filter_state', 'frame_length_s', 'on_threshold_db', 'off_threshold_db', 'dp_s', 'dp_e'])

        csv_title = 'peaks_' + os.path.basename(input_set_path).split('.')[0] + '.csv'
        csv_path = os.path.join(OUTPUT_FOLDER, csv_title)
        df_peak_boundaries.to_csv(csv_path)
        print('Saved peaks at: ',csv_path)