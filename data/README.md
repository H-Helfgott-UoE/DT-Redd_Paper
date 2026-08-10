## Contents

- **DT_redd_detections.csv**: DT-Redd's output on the full study period (see paper) with optimal parameters, including all automatically detected redd diggings. *dp_s* and *dp_e* correspond to the start and end datapoints of each event, which can be converted back to seconds by dividing by file sample rate (here 48kHz). 
- **file_info.json**: List of all files on which DT-Redd was run, and their respective lengths. Used to compute compensated event counts.
- **parameter_search_results_filter4kHz.csv**: Results of parameter search with audio filtering (4kHz high-pass)
- **parameter_search_results_nofilter.csv**: Results of parameter search with no audio filtering
- **raven_annotations.txt**: Manually annotated events for the full study period (see paper)
- **splits_breakdown**: Breakdown of 80/20 data split, and individual folds for the 80% calibration data