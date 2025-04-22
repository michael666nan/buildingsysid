import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, linalg

class IDDataCorrelation:
    """Mixin with timeseries plotting functionality"""
    
    def plot_cross_correlation(self, lag_range=(0,30), figsize=(12, 8), title=None, grid=True, 
                        show=True, normalize=True, max_lag_percentage=0.25):
        """
        Plot the cross-correlation between inputs and outputs using stem plots.
        
        For normalized correlations, 95% confidence interval boundaries are displayed to help identify
        statistically significant correlations.
        
        Args:
            iddata: The IDData object containing the signals to plot
            lag_range (tuple, optional): Range of lags to plot as (min_lag, max_lag).
                                         If None, uses a percentage of the total samples.
            figsize (tuple, optional): Figure size as (width, height)
            title (str, optional): Title for the figure
            grid (bool, optional): Whether to display grid lines
            show (bool, optional): Whether to call plt.show() after creating the plot
            normalize (bool, optional): Whether to normalize the correlation and show 
                                       confidence intervals
            max_lag_percentage (float, optional): Maximum lag as a percentage of total samples
                                                 when lag_range is None
        
        Returns:
            tuple: (fig, axs) - Figure and axes objects for further customization
        """
        # Create subplots - one for each input-output pair
        n_plots = self.n_inputs * self.n_outputs
        fig, axs = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        # Ensure axs is always a list/array even if n_plots=1
        if n_plots == 1:
            axs = [axs]
        
        # Determine lag range if not provided
        if lag_range is None:
            max_lag = int(self.n_samples * max_lag_percentage)
            lag_range = (-max_lag, max_lag)
        
        # Create lag array
        lags = np.arange(lag_range[0], lag_range[1] + 1)
        
        # Plot counter
        plot_idx = 0
        
        # Loop through each input-output pair
        for i in range(self.n_inputs):
            for j in range(self.n_outputs):
                ax = axs[plot_idx]
                
                # Get input and output data
                u_data = self.u[i, :]
                y_data = self.y[j, :]
                
                # Calculate cross-correlation
                correlations = np.zeros(len(lags))
                
                for k, lag in enumerate(lags):
                    if lag >= 0:
                        # Positive lag: y is delayed relative to u
                        if lag >= self.n_samples:
                            correlations[k] = 0
                        else:
                            # Correlation between u[:-lag] and y[lag:]
                            u_shifted = u_data[:-lag] if lag > 0 else u_data
                            y_shifted = y_data[lag:] if lag > 0 else y_data
                            
                            # Make sure they have the same length
                            min_length = min(len(u_shifted), len(y_shifted))
                            u_shifted = u_shifted[:min_length]
                            y_shifted = y_shifted[:min_length]
                            
                            # Calculate correlation coefficient
                            if normalize:
                                # Normalized cross-correlation (Pearson's r)
                                if np.std(u_shifted) > 0 and np.std(y_shifted) > 0:
                                    correlations[k] = np.corrcoef(u_shifted, y_shifted)[0, 1]
                                else:
                                    correlations[k] = 0
                            else:
                                # Raw covariance
                                correlations[k] = np.mean((u_shifted - np.mean(u_shifted)) * 
                                                        (y_shifted - np.mean(y_shifted)))
                    else:
                        # Negative lag: u is delayed relative to y
                        lag_abs = abs(lag)
                        if lag_abs >= self.n_samples:
                            correlations[k] = 0
                        else:
                            # Correlation between u[lag_abs:] and y[:-lag_abs]
                            u_shifted = u_data[lag_abs:] if lag_abs > 0 else u_data
                            y_shifted = y_data[:-lag_abs] if lag_abs > 0 else y_data
                            
                            # Make sure they have the same length
                            min_length = min(len(u_shifted), len(y_shifted))
                            u_shifted = u_shifted[:min_length]
                            y_shifted = y_shifted[:min_length]
                            
                            # Calculate correlation coefficient
                            if normalize:
                                # Normalized cross-correlation (Pearson's r)
                                if np.std(u_shifted) > 0 and np.std(y_shifted) > 0:
                                    correlations[k] = np.corrcoef(u_shifted, y_shifted)[0, 1]
                                else:
                                    correlations[k] = 0
                            else:
                                # Raw covariance
                                correlations[k] = np.mean((u_shifted - np.mean(u_shifted)) * 
                                                        (y_shifted - np.mean(y_shifted)))
                
                # Plot correlation as stem plot (vertical lines with markers)
                markerline, stemlines, baseline = ax.stem(lags, correlations, basefmt='k-')
                plt.setp(markerline, markersize=3, markerfacecolor='blue', markeredgecolor='blue')
                plt.setp(stemlines, linewidth=1, color='blue', alpha=0.7)
                
                # Add title and labels
                if i == 0 and j == 0 and title:
                    ax.set_title(title)
                elif i == 0 and j == 0:
                    ax.set_title('Cross-correlation between Inputs and Outputs')
                
                # Add vertical line at zero lag
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
                
                # Add horizontal line at zero correlation
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                # Set labels in a more compact format
                input_label = self.u_names[i]
                output_label = self.y_names[j]
                if self.u_units[i]:
                    input_label += f" [{self.u_units[i]}]"
                if self.y_units[j]:
                    output_label += f" [{self.y_units[j]}]"
                    
                # More concise label format
                ax.set_ylabel(f"{output_label}\nvs\n{input_label}", fontsize=9)
                
                # Add a text annotation with the maximum correlation and its lag
                max_corr_idx = np.argmax(np.abs(correlations))
                max_corr = correlations[max_corr_idx]
                max_corr_lag = lags[max_corr_idx]
                
                # Improved annotation style with better positioning
                ax.text(0.02, 0.90, f"Max corr: {max_corr:.3f} at lag: {max_corr_lag}",
                       transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, 
                                                        edgecolor='gray', boxstyle='round,pad=0.5'))
                
                # Calculate 95% confidence interval threshold
                # For Pearson correlation, threshold is approximately 2/sqrt(N)
                # where N is the number of effective samples after accounting for lag
                if normalize:
                    # Calculate the effective number of samples (accounting for maximum lag)
                    n_effective = self.n_samples - max(abs(lag_range[0]), abs(lag_range[1]))
                    if n_effective < 3:  # Minimum required for valid CI
                        n_effective = 3
                    
                    # 95% significance threshold (two-tailed test)
                    conf_threshold = 1.96 / np.sqrt(n_effective - 3)
                    
                    # Plot horizontal lines for significance threshold
                    ax.axhline(y=conf_threshold, color='r', linestyle=':', alpha=0.7, 
                              label='95% CI')
                    ax.axhline(y=-conf_threshold, color='r', linestyle=':', alpha=0.7)
                    
                    # Add text annotation for the significance threshold
                    ax.text(0.02, 0.82, f"95% CI: ±{conf_threshold:.3f}",
                          transform=ax.transAxes, fontsize=8,
                          bbox=dict(facecolor='white', alpha=0.8, 
                                   edgecolor='gray', boxstyle='round,pad=0.5'))
                
                if grid:
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                plot_idx += 1
        
        # Set common x-label for bottom plot only
        axs[-1].set_xlabel('Lag (samples)')
        
        # Use a common y-label on the left side
        if normalize:
            fig.text(0.01, 0.5, 'Correlation coefficient (r)', ha='center', va='center', 
                    rotation='vertical', fontsize=10)
        else:
            fig.text(0.01, 0.5, 'Covariance', ha='center', va='center', rotation='vertical', 
                    fontsize=10)
        
        # Adjust layout with more room for y-labels
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, bottom=0.1, hspace=0.3)
        
        # Show the plot if requested
        if show:
            plt.show()
        
        return
    
    
    def plot_correlation_matrix(self, figsize=(10, 8), cmap='coolwarm', title=None, 
                               show=True, absolute=False, annotate=True, include_scatterplots=False,
                               inputs_only=True):
        """
        Plot a correlation matrix between signals.
        
        This function creates a heatmap showing the correlation coefficients
        between signals to visualize their relationships. By default, shows only
        correlations between input signals. Optionally includes scatterplots to
        visualize each relationship.
        
        Args:
            iddata: The IDData object containing the signals to analyze
            figsize (tuple, optional): Figure size as (width, height)
            cmap (str, optional): Colormap to use for the heatmap
            title (str, optional): Title for the figure
            show (bool, optional): Whether to call plt.show() after creating the plot
            absolute (bool, optional): Whether to plot absolute correlation values
            annotate (bool, optional): Whether to annotate the heatmap cells with values
            include_scatterplots (bool, optional): Whether to include scatterplots for each correlation
            inputs_only (bool, optional): Whether to include only input signals (default: True)
            
        Returns:
            tuple: (fig, ax) - Figure and axis objects for further customization
        """
        # Get the number of inputs and outputs
        n_inputs = self.n_inputs
        
        if inputs_only:
            # Use only input signals
            signals = self.u
            signal_names = self.u_names
            n_signals = n_inputs
        else:
            # Combine signals and signal names for processing
            n_outputs = self.n_outputs
            n_signals = n_inputs + n_outputs
            signals = np.vstack((self.u, self.y))
            signal_names = self.u_names + self.y_names
        
        # Calculate the correlation matrix
        correlation_matrix = np.zeros((n_signals, n_signals))
        
        for i in range(n_signals):
            for j in range(n_signals):
                sig_i = signals[i, :]
                sig_j = signals[j, :]
                
                # Calculate correlation
                if np.std(sig_i) > 0 and np.std(sig_j) > 0:
                    correlation_matrix[i, j] = np.corrcoef(sig_i, sig_j)[0, 1]
                else:
                    correlation_matrix[i, j] = 0
        
        # If include_scatterplots is True, create a grid of scatterplots
        if include_scatterplots:
            # Calculate grid size based on number of signals
            fig = plt.figure(figsize=(figsize[0] * 1.5, figsize[1] * 1.5))
            
            # Create a layout with a correlation matrix in the upper left
            # and scatterplots in the remaining space
            gs = plt.GridSpec(n_signals, n_signals, figure=fig)
            
            # Add correlation matrix in the upper left
            ax_mat = fig.add_subplot(gs[:n_signals//2, :n_signals//2])
            
            # If absolute is True, take absolute value of correlations
            displayed_matrix = np.abs(correlation_matrix) if absolute else correlation_matrix
            vmin, vmax = (0, 1) if absolute else (-1, 1)
            
            # Create the heatmap
            im = ax_mat.imshow(displayed_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Correlation coefficient", rotation=-90, va="bottom")
            
            # Set title
            if title:
                ax_mat.set_title(title)
            else:
                matrix_title = "Input Correlation Matrix" if inputs_only else "Signal Correlation Matrix"
                ax_mat.set_title(matrix_title)
            
            # Set labels
            ax_mat.set_xticks(np.arange(n_signals))
            ax_mat.set_yticks(np.arange(n_signals))
            ax_mat.set_xticklabels(signal_names, rotation=45, ha="right", rotation_mode="anchor")
            ax_mat.set_yticklabels(signal_names)
            
            # Annotate heatmap cells with values if requested
            if annotate:
                fmt = '.2f'
                for i in range(n_signals):
                    for j in range(n_signals):
                        value = displayed_matrix[i, j]
                        text_color = "white" if (absolute and value > 0.5) or (not absolute and abs(value) > 0.5) else "black"
                        ax_mat.text(j, i, f"{value:{fmt}}", ha="center", va="center", 
                                  color=text_color, fontsize=8)
            
            # Add scatterplots for selected correlations
            # We'll add scatterplots for all non-diagonal elements
            for i in range(n_signals):
                for j in range(n_signals):
                    if i != j:  # Skip the diagonal
                        ax = fig.add_subplot(gs[i, j])
                        
                        # Get data for scatterplot
                        x_data = signals[j, :]
                        y_data = signals[i, :]
                        
                        # Plot with small dots and alpha for better visibility
                        ax.scatter(x_data, y_data, s=5, alpha=0.6, 
                                  color='blue' if correlation_matrix[i, j] >= 0 else 'red')
                        
                        # Add regression line
                        if np.std(x_data) > 0 and np.std(y_data) > 0:
                            m, b = np.polyfit(x_data, y_data, 1)
                            ax.plot(x_data, m*x_data + b, 'k-', linewidth=1)
                        
                        # Remove ticks for cleaner look
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # Add correlation value as text
                        corr_val = correlation_matrix[i, j]
                        ax.text(0.05, 0.95, f"r = {corr_val:.2f}", transform=ax.transAxes,
                               fontsize=7, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                        
                        # Only add axis labels on the edges
                        if i == n_signals-1:
                            ax.set_xlabel(signal_names[j], fontsize=8)
                        if j == 0:
                            ax.set_ylabel(signal_names[i], fontsize=8)
            
            plt.tight_layout()
            
        else:
            # Create a basic correlation matrix plot
            fig, ax = plt.subplots(figsize=figsize)
            
            # If absolute is True, take absolute value of correlations
            if absolute:
                correlation_matrix = np.abs(correlation_matrix)
                vmin, vmax = 0, 1
            else:
                vmin, vmax = -1, 1
            
            # Create the heatmap
            im = ax.imshow(correlation_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Correlation coefficient", rotation=-90, va="bottom")
            
            # Set title
            if title:
                ax.set_title(title)
            else:
                matrix_title = "Input Correlation Matrix" if inputs_only else "Signal Correlation Matrix"
                ax.set_title(matrix_title)
            
            # Set labels
            ax.set_xticks(np.arange(n_signals))
            ax.set_yticks(np.arange(n_signals))
            ax.set_xticklabels(signal_names)
            ax.set_yticklabels(signal_names)
            
            # Rotate the x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add grid lines
            ax.set_xticks(np.arange(n_signals+1)-.5, minor=True)
            ax.set_yticks(np.arange(n_signals+1)-.5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)
            
            # Annotate heatmap cells with values if requested
            if annotate:
                fmt = '.2f'
                for i in range(n_signals):
                    for j in range(n_signals):
                        value = correlation_matrix[i, j]
                        text_color = "white" if (absolute and value > 0.5) or (not absolute and abs(value) > 0.5) else "black"
                        ax.text(j, i, f"{value:{fmt}}", ha="center", va="center", 
                               color=text_color, fontsize=9)
            
            plt.tight_layout()
        
        # Show the plot if requested
        if show:
            plt.show()
    
     
    def plot_scatter_matrix(self, figsize=(12, 10), title=None, show=True, 
                                  include_trend=True, max_points=1000, inputs_only=True):
        """
        Create a comprehensive grid of pairwise scatter plots between signals.
        
        This function creates a grid of scatter plots for all pairs of signals with 
        correlation coefficients shown on each plot. The diagonal contains histograms 
        showing the distribution of each signal. By default, only includes input signals.
        
        Args:
            iddata: The IDData object containing the signals to analyze
            figsize (tuple, optional): Figure size as (width, height)
            title (str, optional): Title for the figure
            show (bool, optional): Whether to call plt.show() after creating the plot
            include_trend (bool, optional): Whether to include trend lines on scatter plots
            max_points (int, optional): Maximum number of points to plot to avoid overplotting
            inputs_only (bool, optional): Whether to include only input signals (default: True)
            
        Returns:
            tuple: (fig, axes) - Figure and axes objects for further customization
        """
        # Get the number of inputs
        n_inputs = self.n_inputs
        
        if inputs_only:
            # Use only input signals
            signals = self.u
            signal_names = self.u_names
            n_signals = n_inputs
        else:
            # Combine signals and signal names for processing
            n_outputs = self.n_outputs
            n_signals = n_inputs + n_outputs
            signals = np.vstack((self.u, self.y))
            signal_names = self.u_names + self.y_names
        
        # Create the figure and grid of subplots
        fig, axes = plt.subplots(n_signals, n_signals, figsize=figsize)
        
        # If only one signal, make axes a 2D array
        if n_signals == 1:
            axes = np.array([[axes]])
        
        # Set the title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            title_text = 'Input Pairwise Correlation Analysis' if inputs_only else 'Pairwise Correlation Analysis'
            fig.suptitle(title_text, fontsize=16)
        
        # Sample points if needed to avoid overplotting
        n_samples = self.n_samples
        if n_samples > max_points:
            sample_indices = np.sort(np.random.choice(n_samples, max_points, replace=False))
            sampled_signals = signals[:, sample_indices]
        else:
            sampled_signals = signals
        
        # Plot each pair of signals
        for i in range(n_signals):
            for j in range(n_signals):
                ax = axes[i, j]
                
                if i == j:  # Diagonal: histogram of the signal
                    # Get the signal data
                    signal_data = signals[i, :]
                    
                    # Plot histogram with 'auto' bins
                    ax.hist(signal_data, bins='auto', alpha=0.7, color='steelblue')
                    
                    # Add vertical line for mean
                    ax.axvline(np.mean(signal_data), color='r', linestyle='--', linewidth=1)
                    
                    # Add label only on the diagonal
                    ax.set_title(signal_names[i], fontsize=10)
                    
                    # Add mean and std as text
                    mean = np.mean(signal_data)
                    std = np.std(signal_data)
                    ax.text(0.05, 0.95, f"Mean: {mean:.2f}\nStd: {std:.2f}", 
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                                                            facecolor='white', alpha=0.8))
                    
                else:  # Off-diagonal: scatter plot
                    # Get x and y data
                    x_data = sampled_signals[j, :]
                    y_data = sampled_signals[i, :]
                    
                    # Calculate correlation
                    if np.std(x_data) > 0 and np.std(y_data) > 0:
                        corr = np.corrcoef(signals[j, :], signals[i, :])[0, 1]
                    else:
                        corr = 0
                    
                    # Determine color based on correlation
                    if corr >= 0:
                        color = 'blue'
                    else:
                        color = 'red'
                    
                    # Plot scatter with smaller points for better visibility
                    ax.scatter(x_data, y_data, s=5, alpha=0.5, color=color)
                    
                    # Add trend line if requested
                    if include_trend and np.std(x_data) > 0 and np.std(y_data) > 0:
                        m, b = np.polyfit(x_data, y_data, 1)
                        x_range = np.linspace(np.min(x_data), np.max(x_data), 100)
                        ax.plot(x_range, m * x_range + b, 'k-', linewidth=1)
                    
                    # Add correlation coefficient as text
                    ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Clean up axes
                if i < n_signals - 1:  # Remove x tick labels for all but bottom row
                    ax.set_xticklabels([])
                if j > 0:  # Remove y tick labels for all but leftmost column
                    ax.set_yticklabels([])
                    
                # Add x and y labels only on the edges
                if i == n_signals - 1:
                    ax.set_xlabel(signal_names[j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(signal_names[i], fontsize=9)
                
                # Make ticks smaller
                ax.tick_params(axis='both', which='major', labelsize=8)
                
        # Tight layout with room for title
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Show if requested
        if show:
            plt.show()
        
    
    def plot_partial_cross_correlation(self, lag_range=(0,30), figsize=(12, 8), 
                                  title=None, grid=True, show=True):
        """
        Plot the partial cross-correlation between inputs and outputs, conditioning on all other inputs.
        
        For each input-output pair, this function computes the partial cross-correlation by removing
        the linear effects of all other input variables.
        
        Args:
            lag_range (tuple): Range of lags to plot as (min_lag, max_lag)
            figsize (tuple): Figure size as (width, height)
            title (str): Title for the figure
            grid (bool): Whether to display grid lines
            show (bool): Whether to call plt.show() after creating the plot
            
        Returns:
            tuple: (fig, axs) - Figure and axes objects for further customization
        """
        
        # Create subplots - one for each input-output pair
        n_plots = self.n_inputs * self.n_outputs
        fig, axs = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        # Ensure axs is always a list/array even if n_plots=1
        if n_plots == 1:
            axs = [axs]
        
        # Create lag array
        lags = np.arange(lag_range[0], lag_range[1] + 1)
        
        # Set the main figure title
        if title:
            fig.suptitle(title, fontsize=14)
        else:
            fig.suptitle('Partial Cross-Correlation (Controlling for all other inputs)', fontsize=14)
        
        # Plot counter
        plot_idx = 0
        
        # Loop through each input-output pair
        for target_input_idx in range(self.n_inputs):
            for output_idx in range(self.n_outputs):
                ax = axs[plot_idx]
                
                # Prepare labels
                input_name = self.u_names[target_input_idx]
                output_name = self.y_names[output_idx]
                
                # Format units
                if hasattr(self, 'u_units') and len(self.u_units) > target_input_idx and self.u_units[target_input_idx]:
                    input_with_unit = f"{input_name}\n[{self.u_units[target_input_idx]}]"
                else:
                    input_with_unit = input_name
                    
                if hasattr(self, 'y_units') and len(self.y_units) > output_idx and self.y_units[output_idx]:
                    output_with_unit = f"{output_name}\n[{self.y_units[output_idx]}]"
                else:
                    output_with_unit = output_name
                
                # Determine which inputs to condition on - all inputs except the target
                conditioning_idx = [i for i in range(self.n_inputs) if i != target_input_idx]
                conditioning_names = [self.u_names[i] for i in conditioning_idx]
                
                # Calculate partial cross-correlation for each lag
                partial_corr = []
                corr_p_values = []
                
                for lag in lags:
                    # Initialize variables for this lag
                    u_target = None
                    y_lagged = None
                    conditioning_signals = []
                    min_length = 0
                    
                    if lag >= 0:
                        # Positive lag: output is lagged behind input
                        if lag >= self.n_samples:
                            partial_corr.append(0)
                            corr_p_values.append(1)
                            continue
                        
                        # Prepare target signals - y(t+lag) and u(t)
                        u_target = self.u[target_input_idx, :-lag] if lag > 0 else self.u[target_input_idx, :]
                        y_lagged = self.y[output_idx, lag:] if lag > 0 else self.y[output_idx, :]
                        
                        # Make sure they have the same length
                        min_length = min(len(u_target), len(y_lagged))
                        u_target = u_target[:min_length]
                        y_lagged = y_lagged[:min_length]
                        
                        # Skip if not enough data points
                        if min_length < 5:
                            partial_corr.append(0)
                            corr_p_values.append(1)
                            continue
                        
                        # Prepare conditioning signals (all with same alignment as u_target)
                        for idx in conditioning_idx:
                            u_cond = self.u[idx, :-lag] if lag > 0 else self.u[idx, :]
                            u_cond = u_cond[:min_length]  # Ensure same length as target
                            conditioning_signals.append(u_cond)
                    
                    else:
                        # Negative lag: input is lagged behind output
                        lag_abs = abs(lag)
                        if lag_abs >= self.n_samples:
                            partial_corr.append(0)
                            corr_p_values.append(1)
                            continue
                        
                        # Prepare target signals - y(t) and u(t+lag_abs)
                        u_target = self.u[target_input_idx, lag_abs:]
                        y_lagged = self.y[output_idx, :-lag_abs] if lag_abs > 0 else self.y[output_idx, :]
                        
                        # Make sure they have the same length
                        min_length = min(len(u_target), len(y_lagged))
                        u_target = u_target[:min_length]
                        y_lagged = y_lagged[:min_length]
                        
                        # Skip if not enough data points
                        if min_length < 5:
                            partial_corr.append(0)
                            corr_p_values.append(1)
                            continue
                        
                        # Prepare conditioning signals (all with same alignment as u_target)
                        for idx in conditioning_idx:
                            u_cond = self.u[idx, lag_abs:]
                            u_cond = u_cond[:min_length]  # Ensure same length as target
                            conditioning_signals.append(u_cond)
                    
                    # If no conditioning signals (e.g., single input system), use regular correlation
                    if not conditioning_signals:
                        # Just calculate regular correlation
                        corr_coef = np.corrcoef(u_target, y_lagged)[0, 1]
                        partial_corr.append(corr_coef)
                        # Approximate p-value for Pearson correlation
                        t_stat = corr_coef * np.sqrt((min_length - 2) / (1 - corr_coef**2))
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), min_length - 2))
                        corr_p_values.append(p_value)
                        continue
                    
                    # Compute partial correlation
                    try:
                        # Stack all conditioning signals into a matrix
                        X = np.column_stack(conditioning_signals)
                        
                        # Center all variables
                        X_centered = X - np.mean(X, axis=0)
                        u_centered = u_target - np.mean(u_target)
                        y_centered = y_lagged - np.mean(y_lagged)
                        
                        # Residualize u and y on X (remove linear effects of conditioning variables)
                        try:
                            # Try direct least squares
                            beta_u = linalg.lstsq(X_centered, u_centered)[0]
                            beta_y = linalg.lstsq(X_centered, y_centered)[0]
                        except:
                            # Fall back to regularized regression if needed
                            X_with_reg = X_centered.T @ X_centered + 1e-10 * np.eye(X_centered.shape[1])
                            beta_u = linalg.solve(X_with_reg, X_centered.T @ u_centered)
                            beta_y = linalg.solve(X_with_reg, X_centered.T @ y_centered)
                        
                        # Compute residuals
                        u_resid = u_centered - X_centered @ beta_u
                        y_resid = y_centered - X_centered @ beta_y
                        
                        # Partial correlation is correlation between residuals
                        pc = np.corrcoef(u_resid, y_resid)[0, 1]
                        
                        # Compute p-value for partial correlation
                        # Degrees of freedom: n - 2 - number of conditioning variables
                        df = min_length - 2 - len(conditioning_signals)
                        if df > 0:
                            t_stat = pc * np.sqrt(df / (1 - pc**2))
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                        else:
                            p_value = 1.0
                        
                        partial_corr.append(pc)
                        corr_p_values.append(p_value)
                        
                    except Exception as e:
                        print(f"Error at lag {lag}: {e}")
                        partial_corr.append(0)
                        corr_p_values.append(1)
                
                # Convert to numpy arrays
                partial_corr = np.array(partial_corr)
                corr_p_values = np.array(corr_p_values)
                
                # Plot the partial cross-correlation
                markerline, stemlines, baseline = ax.stem(lags, partial_corr, basefmt='k-')
                plt.setp(markerline, markersize=3, markerfacecolor='blue', markeredgecolor='blue')
                plt.setp(stemlines, linewidth=1, color='blue', alpha=0.7)
                
                # Highlight significant correlations
                significant = corr_p_values < 0.05  # 5% significance level
                if np.any(significant):
                    sig_lags = lags[significant]
                    sig_corr = partial_corr[significant]
                    
                    # Create a new stem plot for significant correlations
                    markerline2, stemlines2, baseline2 = ax.stem(sig_lags, sig_corr, basefmt=' ')
                    plt.setp(markerline2, markersize=5, markerfacecolor='red', markeredgecolor='red')
                    plt.setp(stemlines2, linewidth=1.5, color='red', alpha=0.9)
                
                # Add vertical line at zero lag
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
                
                # Add horizontal line at zero correlation
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                # Calculate 95% confidence interval threshold
                n_effective = self.n_samples - max(abs(lag_range[0]), abs(lag_range[1]))
                df = max(n_effective - 2 - len(conditioning_idx), 1)
                conf_threshold = 1.96 / np.sqrt(df)
                
                # Plot confidence intervals
                ax.axhline(y=conf_threshold, color='r', linestyle=':', alpha=0.7, label='95% CI')
                ax.axhline(y=-conf_threshold, color='r', linestyle=':', alpha=0.7)
                
                # Set subplot title
                if conditioning_names:
                    cond_str = ", ".join(conditioning_names)
                    plot_title = f"Controlling for: {cond_str}"
                else:
                    plot_title = f"Regular Cross-Correlation"
                    
                ax.set_title(plot_title, fontsize=9)
                
                # Find maximum correlation and its lag
                if len(partial_corr) > 0:
                    max_idx = np.argmax(np.abs(partial_corr))
                    max_corr = partial_corr[max_idx]
                    max_lag = lags[max_idx]
                    
                    # Add annotation for maximum correlation
                    ax.text(0.02, 0.90, f"Max corr: {max_corr:.3f} at lag: {max_lag}",
                           transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, 
                                                           edgecolor='gray', boxstyle='round,pad=0.5'))
                    
                    # Add annotation for confidence threshold
                    ax.text(0.02, 0.82, f"95% CI: ±{conf_threshold:.3f}",
                           transform=ax.transAxes, fontsize=8,
                           bbox=dict(facecolor='white', alpha=0.8, 
                                    edgecolor='gray', boxstyle='round,pad=0.5'))
                
                # Use multi-line y-axis label format
                y_label = f"Output:\n{output_with_unit}\n\nvs\n\nInput:\n{input_with_unit}"
                ax.set_ylabel(y_label, fontsize=9, linespacing=1.2)
                
                if grid:
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                plot_idx += 1
        
        # Set common x-label for bottom plot only
        axs[-1].set_xlabel('Lag (samples)')
        
        # Add a common y-axis label for the correlation coefficient on the right side
        fig.text(0.99, 0.5, 'Correlation Coefficient', ha='center', va='center', 
                 rotation=270, fontsize=10)  # Note the rotation is now 270 for right side
        
        # Adjust layout with more space on the left for the labels
        plt.tight_layout()
        plt.subplots_adjust(left=0.22, right=0.92, bottom=0.1, top=0.92, hspace=0.3)
        
        if show:
            plt.show()
        
        return