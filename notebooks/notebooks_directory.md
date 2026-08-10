## Contents

- **file_to_frames.ipynb**: For each file in a provided list, outputs a list of frame-by-frame volumes for different frame length settings. Used for parameter searches.
- **frames_to_peaks.ipynb**: From the output of **file_to_frames.ipynb**, computes lists of individual peaks for all possible combinations of provided parameters. Used for parameter searches.
- **make_batches.ipynb**: From folders containing raw audio files, makes batches of custom length for manual annotation and algorithm testing
- **peaks_to_stats.ipynb**: From the output of **frames_to_peaks.ipynb** and a separate manual annotation selection, compute algorithm performance statistics
- **raven_to_stats.ipynb**: Process and compute simple statistics on a manual annotation selection
- **split_files.ipynb**: Given a set of batches, splits files into calibration/validation sets following user-set criteria

All notebooks can be run by adding custom paths and parameters in their respective **Input/Output Files & Parameters** sections.