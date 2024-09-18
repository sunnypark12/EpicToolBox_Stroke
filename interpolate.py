from scipy import interpolate

def interpolate_data(data, time_column='Time', method='linear'):
    """
    Interpolates the given data using the specified method to resample data at uniform time intervals.
    
    Parameters:
    - data: DataFrame containing the kinematic data.
    - time_column: The column name representing time.
    - method: The interpolation method ('linear', 'cubic', etc.).
    
    Returns:
    - interpolated_data: A DataFrame with interpolated values.
    """
    time = data[time_column]
    new_time = np.linspace(time.min(), time.max(), len(time))  # Define a new uniform time grid
    
    interpolated_data = pd.DataFrame({'Time': new_time})
    
    for column in data.columns:
        if column != time_column:
            f = interpolate.interp1d(time, data[column], kind=method, fill_value="extrapolate")
            interpolated_data[column] = f(new_time)
    
    return interpolated_data

# Example Usage
interpolated_df = interpolate_data(data)
print(interpolated_df.head())
