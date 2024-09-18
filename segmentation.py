import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def segment_gait_cycles(data, gc_column, distance_threshold=20):
    """
    Segments trial data by gait cycle using the peaks and troughs of a gait cycle column.
    
    Parameters:
    - data: DataFrame containing the trial data.
    - gc_column: The column that holds the gait cycle information (0-100% GC values).
    - distance_threshold: Minimum distance between peaks and troughs.
    
    Returns:
    - segments: List of dictionaries with start and end times for each gait cycle.
    """
    # Extract the gait cycle column for segmentation
    gc_signal = data[gc_column].values
    
    # Find peaks (local maxima) for the gait cycle signal
    peaks, _ = find_peaks(-gc_signal, distance=distance_threshold)
    
    # Add boundaries (first and last significant points)
    boundaries = np.concatenate(([0], peaks, [len(gc_signal)-1]))
    
    # Compute the intervals between the boundaries
    intervals = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries) - 1)]
    
    # Convert intervals to times using the 'Time' column
    segments = [{"start_time": data['Time'].iloc[start], "end_time": data['Time'].iloc[end]} for start, end in intervals]
    
    return segments


# Example
file_path = "/Users/sunho/Desktop/toolbox_final/SSST10_V1_TM_LG_FD_S2_t1.csv"  # Use your correct file path
data = pd.read_csv(file_path)

# Column containing gait cycle information (e.g., 0-100% GC values)
### TODO : change gc_column to correct gait cycle column)
gc_column = 'back.angular_velocity.y' 

# Segmentation based on gait cycles
segments = segment_gait_cycles(data, gc_column)

# Print Segmented Intervals
for idx, segment in enumerate(segments):
    print(f"Segment {idx+1}: Start Time = {segment['start_time']}, End Time = {segment['end_time']}")
