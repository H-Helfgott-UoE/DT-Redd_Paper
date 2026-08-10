import time
import_start = time.time()
import os
import pickle
import json

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

import soundfile as sf
from tqdm import tqdm

IMPORT_TIME = time.time() - import_start

total_start = time.time()

# ================== INPUT/OUTPUT FILES ===================

ALL_FOLDS = r''
FRAMEVOLUME_PATH = r''
PEAKS_PATH = r''

# ====================== PARAMETERS =======================

FRAME_LENGTH_S = 0.05
OVERLAP = 0.8
CUT_FREQ = 4000

MIN_PEAK_LENGTH = 0.5
MAX_PEAK_LENGTH = 10
MIN_GAP_LENGTH = 0.5

ON_THRESHOLD = 12
OFF_THRESHOLD = 0

SAMPLE_RATE = 48000

# ================= GLOBAL TIME VARIABLES =================

FILE_OP_TIME = 0
FILTER_TIME = 0
TO_FRAMES_TIME = 0
TO_PEAKS_TIME = 0
TOTAL_TIME = 0

# ======================= FUNCTIONS =======================

def power_to_dB(power):
        eps = 1e-12
        return 10 * np.log10(power + eps)

def raw_to_framepower(data, frame_length, step_length):
        frames = sliding_window_view(data, frame_length)[::step_length]
        return np.divide(np.einsum("ij,ij->i", frames, frames), frame_length)

def file_to_framevolumes(file_path, frame_length_s, overlap, cut_freq):

    opening_start = time.time()
    data, samplerate = sf.read(file_path)
    opening_time = time.time() - opening_start

    filter_start = time.time()
    high_pass_filter_sos = signal.butter(4, cut_freq, btype='highpass', fs=samplerate, output='sos')
    data = signal.sosfilt(high_pass_filter_sos, data)
    filter_time = time.time() - filter_start

    global FILE_OP_TIME
    FILE_OP_TIME += opening_time
    global FILTER_TIME
    FILTER_TIME += filter_time

    frame_length = int(frame_length_s * samplerate)
    step_length = int(frame_length * (1 - overlap))

    framevolume = power_to_dB(raw_to_framepower(data, frame_length, step_length))
    return framevolume, opening_time+filter_time

def filelist_to_framevolume_dict(file_list, frame_length_s, overlap, cut_freq):

    start_fvolume = time.time()
    global TO_FRAMES_TIME
    time_fvolume = 0

    filelist_framevolume_dict = {}

    for i, file_info in tqdm(enumerate(file_list), total = len(file_list)):

        sub_folder, filename =  file_info
        file_path = os.path.join(sub_folder, filename)

        framevolume, off_time = file_to_framevolumes(file_path, frame_length_s, overlap, cut_freq)
        time_fvolume = time_fvolume - off_time
        filelist_framevolume_dict[filename] = framevolume

    time_fvolume += time.time() - start_fvolume
    TO_FRAMES_TIME = time_fvolume

    return filelist_framevolume_dict

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

# ===================== MAIN SCRIPT =======================

if __name__ == "__main__":

    # ----------- FILE LIST TO FRAME VOLUMES -----------------------------------
    with open(ALL_FOLDS, "r") as jsonfile: 
        all_folds = json.load(jsonfile)
        file_list = all_folds

    FILE_OP_TIME += time.time() - total_start

    # Select 1440 files: one full day
    filelist_framevolume_dict = filelist_to_framevolume_dict(file_list[:1440], FRAME_LENGTH_S, OVERLAP, CUT_FREQ)

    open_volumes = time.time()
    with open(FRAMEVOLUME_PATH, 'wb') as fp:
        pickle.dump(filelist_framevolume_dict, fp)

    # ----------- FRAME VOLUMES TO PEAK BOUNDARIES -----------------------------

    with open(FRAMEVOLUME_PATH, 'rb') as file:
        frame_volumes_set = pickle.load(file)

    FILE_OP_TIME += time.time() - open_volumes

    peak_start = time.time()

    list_peak_boundaries = []

    # Loop within a set of per-frame volume arrays
    for file_path, frame_volumes in tqdm(frame_volumes_set.items(), total = len(frame_volumes_set)):

        filename = os.path.basename(file_path)

        params = {
            'on_threshold_db':  ON_THRESHOLD,
            'off_threshold_db':  OFF_THRESHOLD,

            'frame_length_s': FRAME_LENGTH_S,
            'overlap': OVERLAP,
            'sample_rate': SAMPLE_RATE,

            'min_peak_length_s': MIN_PEAK_LENGTH,
            'max_peak_length_s': MAX_PEAK_LENGTH,
            'min_gap_s': MIN_GAP_LENGTH
        }

        datapoint_peaks = getPeakBoundaries(frame_volumes, params)

        for dp_s, dp_e in datapoint_peaks:
            list_peak_boundaries.append([filename, CUT_FREQ, str(FRAME_LENGTH_S), ON_THRESHOLD, OFF_THRESHOLD, dp_s, dp_e])

    df_peak_boundaries = pd.DataFrame(list_peak_boundaries, columns = ['filename', 'filter_state', 'frame_length_s', 'on_threshold_db', 'off_threshold_db', 'dp_s', 'dp_e'])

    TO_PEAKS_TIME = time.time() - peak_start

    file_op_last = time.time()

    df_peak_boundaries.to_csv(PEAKS_PATH)

    FILE_OP_TIME += time.time() - file_op_last

    total_end = time.time()

    print('file opening time: ',FILE_OP_TIME)
    print('filtering time ',FILTER_TIME)
    print('framevolumes time: ',TO_FRAMES_TIME)
    print('peaks time: ',TO_PEAKS_TIME)
    print('imports time: ',IMPORT_TIME)
    print('Total time: ',total_end - total_start)