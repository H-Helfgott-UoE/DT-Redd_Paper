import os
import pickle
import json

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

import soundfile as sf
from tqdm import tqdm

# ================== INPUT/OUTPUT FILES ===================

# JSON 2d list of all files in recording period, with format:
# [[folder, filename],
#  [folder, filename]
#  ...
#   ]
FILE_LIST = r''

# Path to file-by-file frame volumes (intermediary results)
FRAMEVOLUME_PATH = r''

# Final output path, with a list of inferred redd digging events
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

# ======================= FUNCTIONS =======================

# Convert a np.array of power values into volumes (dB)
def power_to_dB(power):
        eps = 1e-12
        return 10 * np.log10(power + eps)

# Convert raw audio into a np.array of power values using a sliding window
# of length frame_length and spacing step_length
def raw_to_framepower(data, frame_length, step_length):
        frames = sliding_window_view(data, frame_length)[::step_length]
        return np.divide(np.einsum("ij,ij->i", frames, frames), frame_length)

# Convert raw audio from a given file path to individual frame volumes
def file_to_framevolumes(file_path, frame_length_s, overlap, cut_freq):

    data, samplerate = sf.read(file_path)

    high_pass_filter_sos = signal.butter(4, cut_freq, btype='highpass', fs=samplerate, output='sos')
    data = signal.sosfilt(high_pass_filter_sos, data)

    frame_length = int(frame_length_s * samplerate)
    step_length = int(frame_length * (1 - overlap))

    framevolume = power_to_dB(raw_to_framepower(data, frame_length, step_length))
    return framevolume

# Convert all files in a given folder to frame volumes
def filelist_to_framevolume_dict(file_list, frame_length_s, overlap, cut_freq):

    filelist_framevolume_dict = {}

    for i, file_info in tqdm(enumerate(file_list), total = len(file_list)):

        folder, filename =  file_info
        file_path = os.path.join(folder, filename)

        filelist_framevolume_dict[filename] = file_to_framevolumes(file_path, frame_length_s, overlap, cut_freq)

    return filelist_framevolume_dict

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

# ===================== MAIN SCRIPT =======================

if __name__ == "__main__":

    # ----------- FILE LIST TO FRAME VOLUMES -----------------------------------
    with open(FILE_LIST, "r") as jsonfile: 
        file_list = json.load(jsonfile)

    filelist_framevolume_dict = filelist_to_framevolume_dict(file_list, FRAME_LENGTH_S, OVERLAP, CUT_FREQ)

    with open(FRAMEVOLUME_PATH, 'wb') as fp:
        pickle.dump(filelist_framevolume_dict, fp)
        print('Dictionary saved successfully to file ', FRAMEVOLUME_PATH)

    # ----------- FRAME VOLUMES TO PEAK BOUNDARIES -----------------------------
    print('Processing: ', FRAMEVOLUME_PATH)

    with open(FRAMEVOLUME_PATH, 'rb') as file:
        frame_volumes_set = pickle.load(file)

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

    df_peak_boundaries.to_csv(PEAKS_PATH)
    print('Saved peaks at: ', PEAKS_PATH)