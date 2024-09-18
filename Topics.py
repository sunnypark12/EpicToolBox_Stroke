import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import interpolate

class Topics:
    
    @staticmethod
    def segment_gc(trial_data, gc_topic='gc', gc_channel='gait', distance_threshold=20):
        """
        Segments the trial data by gait cycle using the gait cycle (GC) topic.
        GC must be a triangle wave that holds the 0-100% GC values.
        
        Parameters:
        - trial_data: The full trial dataset containing various topics.
        - gc_topic: The name of the topic containing gait cycle data.
        - gc_channel: The channel within the GC topic holding the gait cycle percentage values.
        - distance_threshold: Minimum distance between peaks for segmentation.
        
        Returns:
        - segments: List of segmented trial data based on gait cycles.
        """
        # Ensure trial_data is a dictionary with the correct structure
        if gc_topic not in trial_data or 'Time' not in trial_data[gc_topic]:
            raise ValueError("The provided trial data does not contain the required gc_topic or 'Time' column.")
        
        gc_data = trial_data[gc_topic][gc_channel]
        intervals = Topics.gc2intervals(gc_data, trial_data[gc_topic]['Time'], distance_threshold)  # Pass time as well for accurate segmentation
        
        # Get all the topics in trial_data to segment
        segments = Topics.segment(trial_data['kinematic_data'], intervals)
        
        return segments


    @staticmethod
    def gc2intervals(gc_data, time_data, distance_threshold):
        """
        Helper function to calculate gait cycle intervals based on peaks.
        
        Parameters:
        - gc_data: Gait cycle percentage data (0-100%).
        - time_data: Time column corresponding to the gait cycle data.
        - distance_threshold: Minimum distance between peaks to avoid false positives.
        
        Returns:
        - intervals: List of time intervals for each gait cycle.
        """
        peaks, _ = find_peaks(-gc_data, distance=distance_threshold)  # Inverting the signal to detect troughs
        boundaries = np.concatenate(([0], peaks, [len(gc_data) - 1]))  # Add start and end points
        
        intervals = [(time_data.iloc[boundaries[i]], time_data.iloc[boundaries[i+1]]) for i in range(len(boundaries) - 1)]
        return intervals


    @staticmethod
    def segment(trial_data, intervals):
        """
        Segments the trial data based on calculated intervals.
        
        Parameters:
        - trial_data: The full dataset to segment.
        - intervals: The intervals calculated from the gait cycle data.
        
        Returns:
        - segmented_data: List of DataFrames, each representing one gait cycle.
        """
        segmented_data = []
        
        # Ensure trial_data is a DataFrame and has the 'Time' column
        if 'Time' not in trial_data.columns:
            raise KeyError("'Time' column not found in the kinematic data.")
        
        # Perform segmentation based on the intervals
        for start, end in intervals:
            segment = trial_data[(trial_data['Time'] >= start) & (trial_data['Time'] <= end)]
            segmented_data.append(segment)
        
        return segmented_data




    @staticmethod
    def interpolate_data(trial_data, time_column='Time', method='linear'):
        """
        Interpolates the given data using the specified method to resample data at uniform time intervals.
        
        Parameters:
        - trial_data: DataFrame containing the kinematic data.
        - time_column: The column name representing time.
        - method: The interpolation method ('linear', 'cubic', etc.).
        
        Returns:
        - interpolated_data: A DataFrame with interpolated values.
        """
        time = trial_data[time_column]
        new_time = np.linspace(time.min(), time.max(), len(time))  # Define a new uniform time grid

        # Dictionary to hold the interpolated columns
        interpolated_columns = {'Time': new_time}

        for column in trial_data.columns:
            if column != time_column:
                f = interpolate.interp1d(time, trial_data[column], kind=method, fill_value="extrapolate")
                interpolated_columns[column] = f(new_time)

        # Create the interpolated DataFrame in one step
        interpolated_data = pd.DataFrame(interpolated_columns)
        
        return interpolated_data




    @staticmethod
    def cut(trial_data, start_time, end_time):
        """
        Cuts the data between two given times.
        
        Parameters:
        - trial_data: DataFrame containing the trial data.
        - start_time: The start time for cutting the data.
        - end_time: The end time for cutting the data.
        
        Returns:
        - cut_data: A DataFrame containing data between start_time and end_time.
        """
        cut_data = trial_data[(trial_data['Time'] >= start_time) & (trial_data['Time'] <= end_time)]
        return cut_data




    @staticmethod
    def normalize(trial_data):
        """
        Normalizes the data in each column between 0 and 1 based on min-max scaling.
        
        Parameters:
        - trial_data: DataFrame containing the trial data.
        
        Returns:
        - normalized_data: A DataFrame with normalized values.
        """
        normalized_data = trial_data.copy()
        for column in trial_data.columns:
            if column != 'Time':  # Don't normalize the time column
                min_val = trial_data[column].min()
                max_val = trial_data[column].max()
                normalized_data[column] = (trial_data[column] - min_val) / (max_val - min_val)
        return normalized_data




    @staticmethod
    def findTimes(condition_func, trial_data):
        """
        Finds the times where a condition is met in the trial data.
        
        Parameters:
        - condition_func: A function that returns a boolean mask on the trial data.
        - trial_data: DataFrame containing the trial data.
        
        Returns:
        - intervals: List of time intervals where the condition is satisfied.
        """
        condition = condition_func(trial_data)
        condition_diff = np.diff(np.concatenate(([0], condition.astype(int))))
        starts = np.where(condition_diff == 1)[0]
        ends = np.where(condition_diff == -1)[0]
        
        if len(ends) < len(starts):
            ends = np.append(ends, len(condition) - 1)
        
        intervals = [(trial_data.iloc[start]['Time'], trial_data.iloc[end]['Time']) for start, end in zip(starts, ends)]
        return intervals




    @staticmethod
    def average(trials_array, topics_list=None):
        """
        Computes the average, standard deviation, max, and min for given topics across multiple trials.
        
        Parameters:
        - trials_array: List of DataFrames, each containing trial data.
        - topics_list: Optional list of topics to compute the statistics for.
        
        Returns:
        - meanvalues, stdvalues, maxvalues, minvalues: DataFrames with computed statistics.
        """
        if topics_list is None:
            topics_list = Topics.topics(trials_array[0])  # Get default topics from the first trial
        
        meanvalues = {}
        stdvalues = {}
        maxvalues = {}
        minvalues = {}

        for topic in topics_list:
            all_messages = []
            skip_topic = False

            for trial in trials_array:
                try:
                    message_data = trial[topic]
                    all_messages.append(message_data.to_numpy())
                except KeyError:
                    print(f"Warning: Topic '{topic}' not found, skipping.")
                    skip_topic = True
                    break

            if skip_topic or len(all_messages) == 0:
                continue

            # If there's only one trial, no need to stack
            if len(all_messages) == 1:
                meanvalues[topic] = all_messages[0]
                stdvalues[topic] = np.zeros_like(all_messages[0])  # No standard deviation for a single trial
                maxvalues[topic] = all_messages[0]
                minvalues[topic] = all_messages[0]
            else:
                # Stack all the message data along the third dimension
                all_messages = np.stack(all_messages, axis=2)

                # Compute the statistics
                meanvalues[topic] = np.nanmean(all_messages, axis=2)
                stdvalues[topic] = np.nanstd(all_messages, axis=2)
                maxvalues[topic] = np.nanmax(all_messages, axis=2)
                minvalues[topic] = np.nanmin(all_messages, axis=2)
        
        return meanvalues, stdvalues, maxvalues, minvalues

    @staticmethod
    def channels(trial_data, topic):
        """
        Returns all channel names in a given topic of the trial data.
        
        Parameters:
        - trial_data: DataFrame containing the trial data.
        - topic: Name of the topic to retrieve channels from.
        
        Returns:
        - allchannels: List of channel names in the given topic.
        """
        try:
            topic_data = trial_data[topic]
        except KeyError:
            print(f"Topic '{topic}' not found in the trial data.")
            return []
        
        # Check if it's a DataFrame or Series
        if isinstance(topic_data, pd.DataFrame):
            return list(topic_data.columns)  # Return column names if DataFrame
        elif isinstance(topic_data, pd.Series):
            return [topic_data.name]  # Return the name of the Series as a single channel
        
        return []  # Return empty if the structure is unexpected


    @staticmethod
    def consolidate(trial_data, topics_list=None, short_names=False, prepend=True):
        """
        Consolidates the channels from different topics into a single table.
        
        Parameters:
        - trial_data: DataFrame or dict containing the trial data.
        - topics_list: Optional list of topics to consolidate.
        - short_names: Whether to use short names for columns.
        - prepend: Whether to prepend the topic name to the column names.
        
        Returns:
        - consolidated: DataFrame with consolidated channels.
        - names: DataFrame mapping short names to long names.
        """
        if topics_list is None:
            topics_list = Topics.topics(trial_data)  # Get all available topics if not provided
        
        feature_table = []
        shortnames = []
        longnames = []
        N = 0  # Running index for feature naming
        
        for topic_name in topics_list:
            try:
                topic_data = trial_data[topic_name]  # Access topic data
            except KeyError:
                print(f"Warning: Topic '{topic_name}' not found, skipping.")
                continue

            # If it's a Series (single column), handle accordingly
            if isinstance(topic_data, pd.Series):
                topic_data = topic_data.to_frame()  # Convert Series to DataFrame

            headers = list(topic_data.columns)  # Get column names
            if prepend:
                headers = [f"{topic_name}_{header}" for header in headers]  # Prepend topic name
            
            # Create short names for the columns
            n = len(headers)
            shortname = [f"F{N + i}" for i in range(1, n + 1)]
            N += n
            
            shortnames.extend(shortname)
            longnames.extend(headers)
            
            feature_table.append(topic_data)

        # Consolidate all data into one DataFrame
        consolidated = pd.concat(feature_table, axis=1)
        names = pd.DataFrame({'ShortName': shortnames, 'LongName': longnames})
        
        if short_names:
            consolidated.columns = shortnames  # Use short names if specified
        else:
            consolidated.columns = longnames  # Otherwise use long names
        
        return consolidated, names


    @staticmethod
    def topics(trial_data):
        """
        Returns a list of all topics in the trial data.
        
        Parameters:
        - trial_data: DataFrame containing the trial data.
        
        Returns:
        - topics_list: List of all topics in the trial data.
        """
        return list(trial_data.keys())
    


# Example usage: load data and test the segmentation

# Load the data
file_path = "/Users/sunho/Desktop/toolbox_final/SSST10_V1_TM_LG_FD_S2_t1.csv"
data = pd.read_csv(file_path)

# Create a trial_data structure as per the Topics class requirements
trial_data = {
    'gc': {
        'gait': data['back.angular_velocity.y'],  # Replace 'gait_cycle_column' with column name for gait cycle
        'Time': data['Time']  # Time column
    },
    'kinematic_data': data 
}


#### Example to segment: ####
# segmented_data = Topics.segment_gc(trial_data, gc_topic='gc', gc_channel='gait', distance_threshold=20)

## Print the segmented data ## 
# for idx, segment in enumerate(segmented_data):
#     print(f"Segment {idx + 1}:")
#     print(segment.head())  # Print first few rows of each segment



#### Example to interpolate: ####
## Perform interpolation on the entire dataset using 'Time' column as reference ## 
# interpolated_data = Topics.interpolate_data(data, time_column='Time', method='linear')

## Print the first few rows of interpolated data ## 
# print(interpolated_data.head())



#### Example to cut: ####
## Define start and end times for cutting the data ## 
# start_time = 1.0  # Set your own start time
# end_time = 3.0    # Set your own end time

## Cut the data within the specified time range ##
# cut_data = Topics.cut(data, start_time, end_time)

## Print the cut data ##
# print(cut_data.head())



#### Example to normalize: ####
## Normalize the data
# normalized_data = Topics.normalize(data)

## Print the normalized data ##
# print(normalized_data.head())



#### Example to find times based on a condition: ####
## Define the threshold and the condition function ## 
# threshold = 0.5
# condition_func = lambda df: df['back.angular_velocity.y'] > threshold

## Find the intervals where the condition is met ## 
# times = Topics.findTimes(condition_func, data)

## Print the intervals ## 
# for idx, interval in enumerate(times):
#     print(f"Interval {idx + 1}: Start Time = {interval[0]}, End Time = {interval[1]}")



#### Example usage for the average function ####
## Create an array of trials, even if it's just one trial ##
# trials_array = [data]  # Add more DataFrames if multiple trials exist

## List of topics to average ##
# topics_list = ['back.angular_velocity.y', 'back.euler_angles.x', 'back.linear_acceleration.x']
## Compute the statistics ##
# meanvalues, stdvalues, maxvalues, minvalues = Topics.average(trials_array, topics_list)

## Print the results ##
# print("Mean values:", meanvalues)
# print("Standard deviation values:", stdvalues)
# print("Max values:", maxvalues)
# print("Min values:", minvalues)



#### Example usage for the channels function ####
# channels = Topics.channels(data, 'back.angular_velocity.y')
# print("Channels:", channels)



#### Example usage for the consolidate function ####
# topics_list = ['back.angular_velocity.y', 'back.euler_angles.x', 'back.linear_acceleration.x']
## Consolidating the data ##
# consolidated_data, column_names = Topics.consolidate(data, topics_list, short_names=False, prepend=True)

## Print consolidated data and column names mapping ##
# print("Consolidated Data:")
# print(consolidated_data.head())
# print("\nColumn Names Mapping:")
# print(column_names)
