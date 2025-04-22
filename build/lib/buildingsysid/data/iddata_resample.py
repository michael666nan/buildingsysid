import datetime
import numpy as np
import copy
import pandas as pd
from scipy import interpolate

class IDDataResample:
    """Mixin with resampling functionality"""

    def resample(self, new_sampling_time=None, method='linear', aggregation=None, 
             output_agg=None, input_agg=None):
        """
        Resample the input and output data to a new sampling time.
        
        Args:
            new_sampling_time (float, optional): New sampling time in seconds. 
                                                If None, returns a copy of the original object.
            method (str, optional): Interpolation method. Options:
                                   'linear' - Linear interpolation (default)
                                   'nearest' - Nearest neighbor interpolation
                                   'cubic' - Cubic spline interpolation
            aggregation (str, optional): Default aggregation method for downsampling. Options:
                                        None - Use interpolation for both up/downsampling (default)
                                        'first' - Use first value in each interval
                                        'last' - Use last value in each interval
                                        'mean' - Use mean value in each interval
                                        'max' - Use maximum value in each interval
                                        'min' - Use minimum value in each interval
            output_agg (str or list, optional): Specific aggregation method(s) for output channels.
                                               Can be a single method or a list with one method per channel.
            input_agg (str or list, optional): Specific aggregation method(s) for input channels.
                                              Can be a single method or a list with one method per channel.
                                   
        Returns:
            IDData: A new IDData object with resampled data
        """
        
        # If no resampling needed, return a copy
        if new_sampling_time is None or new_sampling_time == self.samplingTime:
            return copy.deepcopy(self)
        
        # Determine if this is downsampling (new_sampling_time > old_sampling_time)
        is_downsampling = new_sampling_time > self.samplingTime
        
        # For downsampling with aggregation
        if is_downsampling and (aggregation or output_agg or input_agg):
            return self._downsample_with_aggregation(new_sampling_time, aggregation, output_agg, input_agg)
    
        
        # Get numeric versions of original timestamps
        orig_numeric_ts = self._convert_timestamps_to_numeric(self.timestamps)
        
        # Calculate the new timestamps
        start_time = self.timestamps[0]
        end_time = self.timestamps[-1]
        
        # Create new timestamps array based on the new sampling time
        if isinstance(start_time, pd.Timestamp):
            # For pandas Timestamp
            total_duration = (end_time - start_time).total_seconds()
            num_new_samples = int(total_duration / new_sampling_time) + 1
            new_timestamps = pd.date_range(
                start=start_time,
                periods=num_new_samples,
                freq=pd.Timedelta(seconds=new_sampling_time)
            )
        elif isinstance(start_time, datetime.datetime):
            # For Python datetime
            total_duration = (end_time - start_time).total_seconds()
            num_new_samples = int(total_duration / new_sampling_time) + 1
            new_timestamps = [start_time + datetime.timedelta(seconds=i*new_sampling_time) 
                            for i in range(num_new_samples)]
        elif np.issubdtype(type(start_time), np.datetime64):
            # For numpy datetime64
            total_duration = (end_time - start_time) / np.timedelta64(1, 's')
            num_new_samples = int(total_duration / new_sampling_time) + 1
            time_delta = np.timedelta64(int(new_sampling_time), 's')
            new_timestamps = np.array([start_time + i * time_delta for i in range(num_new_samples)])
        else:
            # For numeric timestamps
            total_duration = end_time - start_time
            num_new_samples = int(total_duration / new_sampling_time) + 1
            new_timestamps = np.linspace(start_time, end_time, num_new_samples)
        
        # Get numeric versions of new timestamps
        new_numeric_ts = self._convert_timestamps_to_numeric(new_timestamps)
        
        # Prepare empty arrays for resampled data
        new_y = np.zeros((self.n_outputs, len(new_timestamps)))
        new_u = np.zeros((self.n_inputs, len(new_timestamps)))
        
        # Resample each output channel
        for i in range(self.n_outputs):
            interpolator = interpolate.interp1d(
                orig_numeric_ts, self.y[i], 
                kind=method, bounds_error=False, fill_value='extrapolate'
            )
            new_y[i] = interpolator(new_numeric_ts)
        
        # Resample each input channel
        for i in range(self.n_inputs):
            interpolator = interpolate.interp1d(
                orig_numeric_ts, self.u[i], 
                kind=method, bounds_error=False, fill_value='extrapolate'
            )
            new_u[i] = interpolator(new_numeric_ts)
        
        # Create a new IDData object with the resampled data
        return self.__class__(
            y=new_y, 
            u=new_u, 
            samplingTime=new_sampling_time,
            timestamps=new_timestamps,
            y_names=self.y_names.copy(),
            u_names=self.u_names.copy(),
            y_units=self.y_units.copy(),
            u_units=self.u_units.copy()
        )

    def _convert_timestamps_to_numeric(self, timestamps):
        """Convert any timestamp format to numeric values for interpolation."""
        if isinstance(timestamps[0], pd.Timestamp):
            # Convert pandas Timestamp to seconds since first timestamp
            first_ts = timestamps[0]
            return np.array([(ts - first_ts).total_seconds() for ts in timestamps])
        elif isinstance(timestamps[0], datetime.datetime):
            # Convert datetime to seconds since first timestamp
            first_ts = timestamps[0]
            return np.array([(ts - first_ts).total_seconds() for ts in timestamps])
        elif np.issubdtype(type(timestamps[0]), np.datetime64):
            # Convert numpy datetime64 to seconds since first timestamp
            first_ts = timestamps[0]
            return np.array([(ts - first_ts) / np.timedelta64(1, 's') for ts in timestamps])
        else:
            # Already numeric
            return np.array(timestamps)

    def _downsample_with_aggregation(self, new_sampling_time, aggregation, output_agg=None, input_agg=None):
        """
        Downsample data using specified aggregation methods.
        
        Args:
            new_sampling_time (float): New sampling time in seconds
            aggregation (str): Default aggregation method for all channels
            output_agg (str or list): Aggregation method(s) for output channels
                                     Can be a single string or a list/tuple with one method per channel
            input_agg (str or list): Aggregation method(s) for input channels
                                    Can be a single string or a list/tuple with one method per channel
            
        Returns:
            IDData: Downsampled data object
        """
        # Set default aggregation method
        default_agg = aggregation or 'mean'
        
        # Process output aggregation methods
        if output_agg is None:
            # Use default for all output channels
            output_agg_methods = [default_agg] * self.n_outputs
        elif isinstance(output_agg, str):
            # Use the same specified method for all output channels
            output_agg_methods = [output_agg] * self.n_outputs
        else:
            # Use provided list of methods
            if len(output_agg) != self.n_outputs:
                print(f"Warning: Length of output_agg ({len(output_agg)}) doesn't match number of outputs ({self.n_outputs}). Using default.")
                output_agg_methods = [default_agg] * self.n_outputs
            else:
                output_agg_methods = output_agg
        
        # Process input aggregation methods
        if input_agg is None:
            # Use default for all input channels
            input_agg_methods = [default_agg] * self.n_inputs
        elif isinstance(input_agg, str):
            # Use the same specified method for all input channels
            input_agg_methods = [input_agg] * self.n_inputs
        else:
            # Use provided list of methods
            if len(input_agg) != self.n_inputs:
                print(f"Warning: Length of input_agg ({len(input_agg)}) doesn't match number of inputs ({self.n_inputs}). Using default.")
                input_agg_methods = [default_agg] * self.n_inputs
            else:
                input_agg_methods = input_agg
        
        # Convert data to DataFrame for easier handling
        output_dfs = []
        for i in range(self.n_outputs):
            df = pd.DataFrame({
                'timestamp': self.timestamps,
                f'y{i}': self.y[i]
            })
            output_dfs.append(df)
        
        input_dfs = []
        for i in range(self.n_inputs):
            df = pd.DataFrame({
                'timestamp': self.timestamps,
                f'u{i}': self.u[i]
            })
            input_dfs.append(df)
        
        # Create resampling rule based on timestamps type
        if isinstance(self.timestamps[0], pd.Timestamp):
            # For pandas timestamps, timestamps are already datetime
            rule = pd.Timedelta(seconds=new_sampling_time)
        elif isinstance(self.timestamps[0], datetime.datetime):
            # Convert Python datetime to pandas datetime
            for df in output_dfs + input_dfs:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            rule = pd.Timedelta(seconds=new_sampling_time)
        elif np.issubdtype(type(self.timestamps[0]), np.datetime64):
            # For numpy datetime64, pandas can handle it directly
            rule = pd.Timedelta(seconds=new_sampling_time)
        else:
            # For numeric timestamps, convert to pandas datetime
            # Assuming numeric timestamps are in seconds from some reference point
            for df in output_dfs + input_dfs:
                # Create an arbitrary date and add timestamp as seconds
                df['timestamp'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(df['timestamp'], unit='s')
            rule = pd.Timedelta(seconds=new_sampling_time)
        
        # Convert aggregation method strings to actual functions
        valid_methods = {'first', 'last', 'mean', 'max', 'min'}
        
        # Resample each output dataframe with its specific aggregation method
        resampled_output_dfs = []
        for i, df in enumerate(output_dfs):
            agg_method = output_agg_methods[i]
            if agg_method not in valid_methods:
                print(f"Warning: Invalid aggregation method '{agg_method}' for output {i}. Using 'mean'.")
                agg_method = 'mean'
                
            df.set_index('timestamp', inplace=True)
            resampled_df = df.resample(rule).agg(agg_method)
            resampled_output_dfs.append(resampled_df)
        
        # Resample each input dataframe with its specific aggregation method
        resampled_input_dfs = []
        for i, df in enumerate(input_dfs):
            agg_method = input_agg_methods[i]
            if agg_method not in valid_methods:
                print(f"Warning: Invalid aggregation method '{agg_method}' for input {i}. Using 'mean'.")
                agg_method = 'mean'
                
            df.set_index('timestamp', inplace=True)
            resampled_df = df.resample(rule).agg(agg_method)
            resampled_input_dfs.append(resampled_df)
        
        # Get new timestamps (all dataframes should have same timestamps after resampling)
        new_timestamps = resampled_output_dfs[0].index
        
        # Convert timestamps back to original format if needed
        if not isinstance(self.timestamps[0], pd.Timestamp):
            if isinstance(self.timestamps[0], datetime.datetime):
                new_timestamps = new_timestamps.to_pydatetime()
            elif np.issubdtype(type(self.timestamps[0]), np.datetime64):
                new_timestamps = new_timestamps.to_numpy()
            else:
                # If original was numeric, convert back to seconds since reference
                new_timestamps = (new_timestamps - pd.Timestamp('2000-01-01')).total_seconds()
        
        # Extract resampled data
        new_y = np.zeros((self.n_outputs, len(new_timestamps)))
        for i, df in enumerate(resampled_output_dfs):
            new_y[i] = df[f'y{i}'].to_numpy()
        
        new_u = np.zeros((self.n_inputs, len(new_timestamps)))
        for i, df in enumerate(resampled_input_dfs):
            new_u[i] = df[f'u{i}'].to_numpy()
        
        # Create a new IDData object with the resampled data
        return self.__class__(
            y=new_y, 
            u=new_u, 
            samplingTime=new_sampling_time,
            timestamps=new_timestamps,
            y_names=self.y_names.copy(),
            u_names=self.u_names.copy(),
            y_units=self.y_units.copy(),
            u_units=self.u_units.copy()
        )