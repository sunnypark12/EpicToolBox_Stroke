import matplotlib.pyplot as plt
from topics_module import Topics
import numpy as np

class Plot:
    @staticmethod
    def plot_imu(trial_data, phases=False, segments=['imu.foot', 'imu.shank', 'imu.thigh'], channelsE=['Roll', 'Pitch', 'Yaw'], channelsQ=['X', 'Y', 'Z', 'W']):
        """
        Plots IMU data from foot, shank, and thigh for given channels.

        Parameters:
        - trial_data: The experiment data with IMU readings.
        - phases: Boolean to indicate if phases should be plotted.
        - segments: List of IMU topics to plot (e.g., foot, shank, thigh).
        - channelsE: List of Euler angle channels to plot.
        - channelsQ: List of quaternion channels to plot.
        """
        num_segments = len(segments)
        fig, axs = plt.subplots(num_segments, 1, figsize=(10, 8))

        for i, segment in enumerate(segments):
            try:
                segment_data = trial_data[segment]
            except KeyError:
                raise KeyError(f"Topic '{segment}' does not exist in the provided trial data.")

            # Plot Quaternion Channels
            for channelQ in channelsQ:
                t = segment_data['Quaternion']['Header']
                x = segment_data['Quaternion'][channelQ]
                axs[i].plot(t, x, label=channelQ)
            
            axs[i].set_title(segment)
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel('Quaternion')
            axs[i].legend()

            if phases and 'fsm' in trial_data:
                Topics.plot_phases(trial_data)
        
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_kinematics(trial_data, phases=False, joints=['ankle', 'knee'], channels=['Theta']):
        """
        Plots the kinematic data for specified joints and channels.

        Parameters:
        - trial_data: The experiment data with joint readings.
        - phases: Boolean to indicate if phases should be plotted.
        - joints: List of joints to plot (e.g., ankle, knee).
        - channels: List of channels to plot (e.g., Theta, ThetaDot).
        """
        num_joints = len(joints)
        fig, axs = plt.subplots(num_joints, 1, figsize=(10, 8))

        for i, joint in enumerate(joints):
            joint_data = trial_data[joint]
            legend_str = []
            ylabel_str = []
            
            for channel in channels:
                t = joint_data['joint_state']['Header']
                x = joint_data['joint_state'][channel]
                axs[i].plot(t, x, label=channel)
                
                if channel == 'Theta':
                    ylbl = 'Angle [deg]'
                elif channel == 'ThetaDot':
                    ylbl = 'Angular vel [deg/s]'
                else:
                    ylbl = 'Unknown'

                legend_str.append(channel)
                ylabel_str.append(ylbl)
            
            axs[i].set_title(f'{joint} joint kinematics')
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel(' / '.join(ylabel_str))
            axs[i].legend(legend_str)

            if phases and 'fsm' in trial_data:
                Topics.plot_phases(trial_data)
        
        plt.tight_layout()
        plt.show()

    

    @staticmethod
    def plot_loadcell(trial_data, phases=False, topic='loadcell.load_cell_readings'):
        """
        Plots loadcell force and moment data.

        Parameters:
        - trial_data: The experiment data with loadcell readings.
        - phases: Boolean to indicate if phases should be plotted.
        - topic: The topic for the loadcell readings.
        """
        try:
            data = trial_data[topic]
        except KeyError:
            raise KeyError(f"Topic '{topic}' does not exist in the provided trial data.")

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Force Data
        t = data['Header']
        axs[0].plot(t, data['ForceX'], label='Fx')
        axs[0].plot(t, data['ForceY'], label='Fy')
        axs[0].plot(t, data['ForceZ'], label='Fz')
        axs[0].set_title('Loadcell Forces')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Force [N]')
        axs[0].legend()

        # Plot Moment Data
        axs[1].plot(t, data['MomentX'], label='Mx')
        axs[1].plot(t, data['MomentY'], label='My')
        axs[1].plot(t, data['MomentZ'], label='Mz')
        axs[1].set_title('Loadcell Moments')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Moment [Nm]')
        axs[1].legend()

        if phases and 'fsm' in trial_data:
            Topics.plot_phases(trial_data)

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_phases(trial_data, style='shades'):
        """
        Plots phases as shaded regions or lines.

        Parameters:
        - trial_data: The experiment data containing phases information.
        - style: 'shades' or 'lines' for the phase visualization.
        """
        states = trial_data['fsm']['State']
        time = states['Header']
        unique_states = np.unique(states['state_column'])  # Replace 'state_column' with actual state data

        fig, ax = plt.subplots()
        Ylim = ax.get_ylim()

        for state in unique_states:
            isphase = (states['state_column'] == state)  # Replace 'state_column' accordingly
            entering = np.where(np.diff(np.concatenate([[0], isphase])) == 1)[0]
            leaving = np.where(np.diff(np.concatenate([[0], isphase])) == -1)[0]
            
            if len(leaving) < len(entering):
                leaving = np.append(leaving, len(isphase) - 1)

            for start, end in zip(entering, leaving):
                t_entering = time[start]
                t_leaving = time[end]

                if style == 'lines':
                    ax.axvline(t_entering, color='b')
                elif style == 'shades':
                    ax.axvspan(t_entering, t_leaving, color='b', alpha=0.3)

        plt.show()


    
    @staticmethod
    def plot_power(trial_data, phases=True, joints=['ankle', 'knee'], channels=['Theta']):
        """
        Plots power data for specified joints.

        Parameters:
        - trial_data: The experiment data containing joint states.
        - phases: Boolean to indicate if phases should be plotted.
        - joints: List of joints to plot.
        - channels: List of channels to plot.
        """
        num_joints = len(joints)
        fig, axs = plt.subplots(num_joints, 1, figsize=(10, 8))

        for i, joint in enumerate(joints):
            joint_data = trial_data[joint]
            for channel in channels:
                t = joint_data['joint_state']['Header']
                x = joint_data['joint_state'][channel]
                axs[i].plot(t, x, label=channel)

                if channel == 'Theta':
                    ylbl = 'Angle [deg]'
                elif channel == 'ThetaDot':
                    ylbl = 'Angular vel [deg/s]'
                else:
                    ylbl = 'Unknown'

            axs[i].set_title(f'{joint} power')
            axs[i].set_xlabel('Time [s]')
            axs[i].set_ylabel(ylbl)
            axs[i].legend()

            if phases and 'fsm' in trial_data:
                Topics.plot_phases(trial_data)
        
        plt.tight_layout()
        plt.show()



    @staticmethod
    def plot_torques(trial_data):
        """
        Plots torque setpoints for ankle and knee.

        Parameters:
        - trial_data: The experiment data containing torque setpoints for joints.
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        ankle_data = trial_data['ankle']['torque_setpoint']
        knee_data = trial_data['knee']['torque_setpoint']

        axs[0].plot(ankle_data['Header'], ankle_data['Torque_'], label='Ankle Torque')
        axs[0].set_title('Ankle Torque')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Torque [Nm]')
        axs[0].legend()

        # Plot Knee Torque
        axs[1].plot(knee_data['Header'], knee_data['Torque_'], label='Knee Torque')
        axs[1].set_title('Knee Torque')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Torque [Nm]')
        axs[1].legend()

        if 'fsm' in trial_data:
            Topics.plot_phases(trial_data)

        plt.tight_layout()
        plt.show()

    
    @staticmethod
    def plot_emg(trial_data, segments=['emg'], channels=['EMG1', 'EMG2', 'EMG3'], phases=False):
        """
        Plots EMG data for the given segments and channels.

        Parameters:
        - trial_data: Dictionary containing trial data.
        - segments: List of segment names in trial_data to plot EMG data.
        - channels: List of channel names corresponding to EMG data in each segment.
        - phases: Boolean indicating whether to overlay phase information on the plot.
        """

        num_segments = len(segments)
        fig, axs = plt.subplots(num_segments, 1, figsize=(10, 6), sharex=True)

        if num_segments == 1:
            axs = [axs]

        for i, segment in enumerate(segments):
            try:
                segment_data = trial_data[segment]
            except KeyError:
                print(f"Error: Segment '{segment}' does not exist in the trial data.")
                continue

            for channel in channels:
                if channel in segment_data.columns:
                    axs[i].plot(segment_data['Time'], segment_data[channel], label=channel)
                else:
                    print(f"Warning: Channel '{channel}' not found in segment '{segment}'.")

            axs[i].set_title(f'{segment} - EMG Data')
            axs[i].set_ylabel('EMG Signal (mV)')
            axs[i].legend()

            if phases and 'fsm' in trial_data:
                Topics.plot_phases(trial_data)

        axs[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        plt.show()



# ### EXAMPLE USAGE ####
# # Example usage of IMU plot ##
# Plot.plot_imu(trial_data, phases=True, segments=['imu.foot', 'imu.shank', 'imu.thigh'], channelsQ=['X', 'Y', 'Z'])

# # Example usage of kinematics plot ##
# Plot.plot_kinematics(trial_data, phases=False, joints=['ankle', 'knee'], channels=['Theta', 'ThetaDot'])

# # Example usage of loadcell plot ##
# Plot.plot_loadcell(trial_data, phases=False)

# # Example usage of power plot ##
# Plot.plot_power(trial_data, phases=True, joints=['ankle', 'knee'], channels=['Theta'])

# # Example usage of torque plot ##
# Plot.plot_torques(trial_data)

# # Example usage of EMG ##
# Example trial_data structure
# trial_data = {
#     'emg': {
#         'Time': data['Time'],
#         'EMG1': data['emg.channel1'],  # Replace with the actual column names
#         'EMG2': data['emg.channel2'],
#         'EMG3': data['emg.channel3']
#     },
#     'fsm': data['fsm']  # Include 'fsm' for phases if available
# }

# # Plot EMG data
# Plot.plot_emg(trial_data, segments=['emg'], channels=['EMG1', 'EMG2', 'EMG3'], phases=True)