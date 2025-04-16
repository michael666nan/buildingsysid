import numpy as np
import matplotlib.pyplot as plt

from .simulate_statespace import simulation



class BaseBlackBox:
    
    def __init__(self, iddata=None, kstep=-999, sum_hor=False):

        self.iddata = iddata
        
        # Cost Function Settings
        self.kstep = kstep      # Zero or less means simulation (No feedback)
        self.sum_hor = sum_hor
        
        # Number of Free Parameters
        self.n_parameters = None
        self.n_states = None
        
        # Bounds on Parameters
        self.lower_bounds = None
        self.upper_bounds = None
        self.lower_bounds_feedback = None
        self.upper_bounds_feedback = None
        
        #Estimation Results
        self.ss = None
        self.par = None
        self.conf_intervals = None
        self.residuals = None
        self.jacobian = None
        self.least_squares_report = None
    

    # =================================================================
    # Create Discrete Time State Space Model (Must be implemented in Child Class)
    # =================================================================
    def create_model(self, par):

        raise NotImplementedError("Subclasses must implement create_model method")
     
        
        
    # =================================================================
    # Create K-matrix
    # =================================================================
    def feedback_matrix(self, par):
        
        if self.kstep>0 and len(par) == (self.n_parameters + self.n_states):
            K = np.array([[par[self.n_parameters + i]] for i in range(self.n_states)])
        else:
            K = np.zeros((self.n_states, 1))
    
        return K

    
    
    # =================================================================
    # Calculate Residuals Vector
    # =================================================================
    def objective(self, par):
        # Create Model
        ss = self.create_model(par)
        
        y_sim = simulation(ss, self.iddata, kstep=self.kstep, sum_horizon=self.sum_hor)
     
        
        # In case of summarized horison --> measured output must be transformed
        if self.sum_hor and self.kstep>0:
            Y = self.iddata.hankel(self.kstep)
            y_mea = Y.T.flatten().reshape((1,-1))
        else:
            y_mea = self.iddata.y
        
        return y_mea[0,:] - y_sim[0,:]



    # =================================================================
    # Calculate Residuals Vector
    # =================================================================
    def _generate_bounds(self):
        self.lower_bounds = np.full(self.n_parameters, -np.inf)
        self.upper_bounds = np.full(self.n_parameters, np.inf)
        self.lower_bounds_feedback = np.full(self.n_states, -np.inf)
        self.upper_bounds_feedback = np.full(self.n_states, np.inf)
    
        # self.lower_bounds = np.full(self.n_parameters, -1)
        # self.upper_bounds = np.full(self.n_parameters, 1)
        # self.lower_bounds_feedback = np.zeros(self.n_states)
        # self.upper_bounds_feedback = np.full(self.n_states, np.inf)
    

    # =================================================================
    # Get Number of Parameters
    # =================================================================
    def get_n_parameters(self):
        
        if self.kstep>0:
            return self.n_parameters + self.n_states
        else:
            return self.n_parameters


    # =================================================================
    # Get Number of Parameters
    # =================================================================
    def get_bounds(self):
         
        if self.kstep>0:
            lb = np.append(self.lower_bounds,self.lower_bounds_feedback)
            ub = np.append(self.upper_bounds,self.upper_bounds_feedback) 
            return (lb, ub)
        else:
            return (self.lower_bounds, self.upper_bounds)


    # =================================================================
    # Print Parameters
    # =================================================================
    def print(self):
        """Print all parameters without scaling and with confidence intervals if provided.
        """
        if not hasattr(self, 'param_dict'):
            # Fallback if param_dict is not defined
            for i, p in enumerate(self.par):
                print(f"Parameter {i}: {p:.2f}")
                if self.conf_intervals is not None and i < self.conf_intervals.shape[1]:
                    lower, upper = self.conf_intervals[0, i], self.conf_intervals[1, i]
                    print(f"    95% CI: [{lower:.2f}, {upper:.2f}]")
            return
            
        # Print model parameters
        for idx, param_info in self.param_dict.items():
            if idx < len(self.par):
                name, unit = param_info[0], param_info[1]
                print(f"{name}: {self.par[idx]:.2f} {unit}")
                
                # Print confidence interval if available
                if self.conf_intervals is not None and idx < self.conf_intervals.shape[1]:
                    lower, upper = self.conf_intervals[0, idx], self.conf_intervals[1, idx]
                    print(f"    95% CI: [{lower:.2f}, {upper:.2f}] {unit}")
        
        # Print feedback parameters if present
        if self.kstep>0 and len(self.par) > self.n_parameters:
            for i in range(self.n_parameters, len(self.par)):
                print(f"k{i - self.n_parameters + 1}: {self.par[i]:.2f}")
                if self.conf_intervals is not None and i < self.conf_intervals.shape[1]:
                    lower, upper = self.conf_intervals[0, i], self.conf_intervals[1, i]
                    print(f"    95% CI: [{lower:.2f}, {upper:.2f}]")
    
    
    # =================================================================
    # Plot Parameters and Confidence Intervals
    # ================================================================= 
    def plot_parameters(self, figsize=(10, 8)):
        """Plot unscaled parameters with their confidence intervals.
        
        Args:
            par: Parameter array
            conf_intervals: Optional confidence intervals as a numpy array with shape (2, n_parameters)
                           where [0, :] contains lower bounds and [1, :] contains upper bounds
            figsize: Figure size as tuple (width, height)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.par.any()==None:
            print("Missing Estimated Parameter(s)")
            return
        
        if not hasattr(self, 'param_dict'):
            # Fallback if param_dict is not defined
            param_names = [f"Parameter {i}" for i in range(len(self.par))]
            param_values = self.par
            param_indices = list(range(len(self.par)))
        else:
            # Extract names from param_dict
            param_names = []
            param_indices = []  # Store original indices
            units = []
            
            for idx in sorted(self.param_dict.keys()):
                if idx < len(self.par):
                    name, unit = self.param_dict[idx][0], self.param_dict[idx][1]
                    param_names.append(name)
                    param_indices.append(idx)  # Store original index
                    units.append(unit)
            
            # Include feedback parameters if present
            if self.kstep>0 and len(self.par) > self.n_parameters:
                for i in range(self.n_parameters, len(self.par)):
                    param_names.append(f"k{i - self.n_parameters + 1}")
                    param_indices.append(i)  # Store original index
                    units.append("")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(param_names))
        
        # Get the unscaled parameter values in the correct order
        param_values = [self.par[idx] for idx in param_indices]
        
        # Plot the parameter values
        bars = ax.bar(x, param_values, width=0.5, color='skyblue', alpha=0.7)
        
        # Plot confidence intervals if provided
        if self.conf_intervals is not None and self.conf_intervals.shape[1] > 0:
            for i, original_idx in enumerate(param_indices):
                if original_idx < self.conf_intervals.shape[1]:
                    # Get unscaled confidence intervals
                    lower = self.conf_intervals[0, original_idx]
                    upper = self.conf_intervals[1, original_idx]
                    
                    # Draw error bars
                    ax.plot([i, i], [lower, upper], 'r-', linewidth=2)
                    ax.plot([i-0.1, i+0.1], [lower, lower], 'r-', linewidth=2)
                    ax.plot([i-0.1, i+0.1], [upper, upper], 'r-', linewidth=2)
        
        # Add value labels on the bars
        for bar, value in zip(bars, param_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Set axis labels and title
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Value')
        ax.set_title('Parameter Estimates with 95% Confidence Intervals')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        
        # Add units to the parameter names in the legend if available
        if 'units' in locals() and len(units) == len(param_names):
            handles = [plt.Rectangle((0,0),1,1, color='skyblue', alpha=0.7)]
            labels = ['Parameter Value']
            ax.legend(handles, labels, loc='upper right')
            
            # Create a secondary y-axis for the units
            ax2 = ax.twinx()
            ax2.set_yticks([])
            
            # Add unit text annotations
            for i, (name, unit) in enumerate(zip(param_names, units)):
                if unit:  # Only add if unit is not empty
                    ax.annotate(f"[{unit}]", xy=(i, 0), xytext=(i, -0.5), 
                                ha='center', va='top', fontsize=8)
        
        plt.tight_layout()
        return fig