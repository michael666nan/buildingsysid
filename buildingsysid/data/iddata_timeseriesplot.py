import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class IDDataTimeSeriesPlot:
    """Mixin with timeseries plotting functionality"""

    def plot_timeseries(self, figsize=(12, 8), time_unit='s', use_timestamps=True, 
                       title=None, grid=True, subplot_kwargs=None, show=True):
        """
        Plot the output and input signals of the system over time.
        
        Outputs are shown as line plots with markers at each sample point.
        Inputs are shown as step plots (zero-order hold) with markers at each sample point.
        
        Args:
            figsize (tuple, optional): Figure size as (width, height)
            time_unit (str, optional): Unit for time axis
            use_timestamps (bool, optional): Whether to use timestamps or sample indices for x-axis
            title (str, optional): Title for the figure
            grid (bool, optional): Whether to display grid lines
            subplot_kwargs (dict, optional): Additional arguments to pass to plt.subplots
            show (bool, optional): Whether to call plt.show() after creating the plot
        
        Returns:
            tuple: (fig, axs) - Figure and axes objects for further customization
        """
        if subplot_kwargs is None:
            subplot_kwargs = {}
        
        # Calculate the number of subplots needed
        n_plots = self.n_outputs + self.n_inputs
        
        # Create a figure with subplots
        fig, axs = plt.subplots(n_plots, 1, figsize=figsize, 
                               sharex=True, **subplot_kwargs)
        
        # Ensure axs is always a list/array even if n_plots=1
        axs = np.atleast_1d(axs)
        
        # Prepare x-axis data
        if use_timestamps and isinstance(self.timestamps[0], (datetime, np.datetime64)):
            # For datetime objects
            x = self.timestamps
            xlabel = 'Time'
        elif use_timestamps:
            # For numeric timestamps
            x = self.timestamps
            xlabel = f'Time [{time_unit}]'
        else:
            # Use sample indices
            x = np.arange(self.n_samples)
            xlabel = 'Sample'
        
        # Plot outputs as lines with markers
        for i in range(self.n_outputs):
            ax = axs[i]
            y_data = self.y[i, :]
            
            # Plot the output data as a continuous line with markers
            ax.plot(x, y_data, 'b-', linewidth=1.5)  # Line
            ax.plot(x, y_data, 'bo', markersize=3)   # Points
            
            # Label with name and unit if provided
            label = self.y_names[i]
            if self.y_units[i]:
                label += f" [{self.y_units[i]}]"
            
            ax.set_ylabel(label)
            
            if i == 0 and title:
                ax.set_title(title)
            elif i == 0:
                ax.set_title('System Outputs and Inputs')
            
            if grid:
                ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot inputs as steps with markers
        for i in range(self.n_inputs):
            ax = axs[i + self.n_outputs]
            u_data = self.u[i, :]
            
            # For step plot, we need to append an extra x value and reuse the last y value
            # to ensure the step extends to the end of the plot
            if len(x) > 1:
                # Calculate the step size to extend beyond the last point
                if isinstance(x[0], (datetime, np.datetime64)):
                    # For datetime x-axis, we need a different approach
                    # Use the average time delta between points to extend
                    time_deltas = np.diff(x)
                    if len(time_deltas) > 0:
                        avg_delta = np.mean(time_deltas)
                        x_extended = np.append(x, x[-1] + avg_delta)
                    else:
                        # Default extension if we can't calculate delta
                        x_extended = np.append(x, x[-1])
                else:
                    # For numeric x-axis
                    if len(x) > 1:
                        step = x[1] - x[0]
                        x_extended = np.append(x, x[-1] + step)
                    else:
                        x_extended = np.append(x, x[-1] + 1)
                
                u_extended = np.append(u_data, u_data[-1])
                
                # Plot the input data as steps
                ax.step(x_extended, u_extended, 'r-', linewidth=1.5, where='post')
                
                # Add points at each actual data point
                ax.plot(x, u_data, 'ro', markersize=3)
            else:
                # If only one point, just plot the point
                ax.plot(x, u_data, 'ro', markersize=5)
            
            # Label with name and unit if provided
            label = self.u_names[i]
            if self.u_units[i]:
                label += f" [{self.u_units[i]}]"
            
            ax.set_ylabel(label)
            
            if grid:
                ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-label on the bottom subplot
        axs[-1].set_xlabel(xlabel)
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the plot if requested
        if show:
            plt.show()
        
        return fig, axs