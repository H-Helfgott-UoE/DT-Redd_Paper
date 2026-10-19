# DT-Redd

This repository contains the code used to conduct all testing, calibration and runs of the DT-Redd algorithm for detecting Atlantic salmon redd digging events from long-term audio data. See associated paper: WILL ADD LINK WHEN PUBLISHED.

## Contents

- **data** includes file info, splits information, parameter search results and both manual and automatic detections for the paper's study period.
- **notebooks** includes jupyter notebooks for all steps of data processing and automatic detection
- **scripts** includes python scripts used to run parameter searches and full detection runs on the paper's study period. Associated scripts and notebooks are identical in content.
A more detailed file-by-file breakdown is available in each folder
- **DT-Redd.py**: Executable DT-Redd algorithm

## How to use

Either keep the default parameters, or edit them by directly modifying DT_Redd.py's *Parameters* section. Then run:

```
python3 DT-Redd.py file_list output_file sample_rate
```

With:
- **file_list** a pre-made json file containing a 2d array of raw audio .WAV file paths in format [[directory, filename], [directory, filename],…]
- **output_file** the path to an output file to store automatically detected events
- **sample_rate** the sample rate in Hz of provided raw audio files

Example:
```
python3 DT-Redd.py file_list.json redd_digging_events.csv 48000
```

The program assumes all audio is provided as .WAV files of identical sample rate, and a file length of 50sec-1min. Results may degrade if provided audio files are significantly shorter or longer.

## Requirements

Python >= 3.14

Libraries (libraries needed for the execution of DT-Redd are indicated in **bold**):
- **os**
- shutil
- pickle
- **json**
- **numpy**
- **pandas**
- scikit-maad
- **scipy.signal**
- **tqdm**
- **soundfile**
- metamoth
