- **file_to_frames.py**: For each file in a provided list, outputs a list of frame-by-frame volumes for different frame length settings. Used for parameter searches.
- **frames_to_peaks.py**: From the output of **file_to_frames.py**, computes lists of individual peaks for all possible combinations of provided parameters. Used for parameter searches.
- **period_to_peaks.py**: Combines the steps of **file_to_frames.py** and **frames_to_peaks.py** for a single combination of parameters. Used for the full run of DT-Redd on the study period
- **peaks_to_stats.py**: From the output of **frames_to_peaks.py** and a separate manual annotation selection, compute algorithm performance statistics
- **time_performance.py**: Copy of **period_to_peaks.py** with added time statistics computations. Used for the speed performance assessment of DT-Redd.

All scripts can be run by adding custom paths and parameters in their respective *Input/Output Files & Parameters* sections.