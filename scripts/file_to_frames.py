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

# Path to folder containing audio file batches
BASE_FOLDER = r''

# Path to json file containing each split and its respective file list 
SPLIT_FILE = r''

OUTPUT_FOLDER = r''

# ====================== PARAMETERS =======================

frame_length_s_list = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
overlap = 0.8
CUT_FREQ = 4000

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

# Convert raw audio from a given file path to individual frame volumes, with one
# separate np.array for each possible frame length value
def file_to_framevolume_dict(file_path, frame_length_s_list, overlap, cut_freq):

    data, samplerate = sf.read(file_path)

    # Replace data with its filtered equivalent if needed
    if cut_freq > 0:
        high_pass_filter_sos = signal.butter(4, cut_freq, btype='highpass', fs=samplerate, output='sos')
        data = signal.sosfilt(high_pass_filter_sos, data)

    framevolume_dict = {}

    # Compute frame volumes according to all provided values of frame_length
    for frame_length_s in frame_length_s_list:

        frame_length = int(frame_length_s * samplerate)
        step_length = int(frame_length * (1 - overlap))

        framevolume = power_to_dB(raw_to_framepower(data, frame_length, step_length))
        framevolume_dict[str(frame_length_s)] = framevolume
    
    return framevolume_dict

# Convert all files in a given folder to frame volumes
def filelist_to_framevolume_dict(base_folder, file_list, frame_length_s_list, overlap, cut_freq):

    filelist_framevolume_dict = {}

    for i, file_info in tqdm(enumerate(file_list), total = len(file_list)):

        split_folder, filename =  file_info
        within_split_path = os.path.join(split_folder, filename)
        file_path = os.path.join(base_folder, within_split_path)

        filelist_framevolume_dict[within_split_path] = file_to_framevolume_dict(file_path, frame_length_s_list, overlap, cut_freq)

    return filelist_framevolume_dict

# ===================== MAIN SCRIPT =======================

if __name__ == "__main__":

    with open(SPLIT_FILE, "r") as jsonfile: 
        all_splits = json.load(jsonfile)

    folds = ['train_fold_0', 'train_fold_1', 'train_fold_2', 'train_fold_3', 'train_fold_4']

    # Fold-by-fold audio conversion to frame volumes, with no filter
    for fold in folds:
        print('Computing frames for: ',fold)
        file_list = all_splits[fold]
        filelist_framepower_dict = filelist_to_framevolume_dict(BASE_FOLDER, file_list, frame_length_s_list, overlap, 0)

        output_name = 'all_frame_volumes_'+fold+'_nofilter.pkl'
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        with open(output_path, 'wb') as fp:
            pickle.dump(filelist_framepower_dict, fp)
            print('Dictionary saved successfully to file ',output_path)

    # Fold-by-fold audio conversion to frame volumes, with high-pass filter of frequency CUTOFF
    for fold in folds:
        print('Computing frames for: ',fold)
        file_list = all_splits[fold]
        filelist_framepower_dict = filelist_to_framevolume_dict(BASE_FOLDER, file_list, frame_length_s_list, overlap, CUT_FREQ)

        output_name = 'all_frame_volumes_'+fold+'_filter4kHz.pkl'
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        with open(output_path, 'wb') as fp:
            pickle.dump(filelist_framepower_dict, fp)
            print('Dictionary saved successfully to file ',output_path)