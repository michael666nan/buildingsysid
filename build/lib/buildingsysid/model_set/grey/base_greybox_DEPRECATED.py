import numpy as np
import matplotlib.pyplot as plt

from buildingsysid.utils.simulate_statespace import simulation

class BaseGreyBox:
    
    def __init__(self, floor_area=1, fixed_params=None):
        self.iddata = None        
        self.floor_area = floor_area
        
        # Cost Function Settings
        self.kstep = -1
        self.sum_hor = False
        
        # Store fixed params for later
        self._fixed_params = fixed_params or {}
        
        # Call the parameter setup method (to be implemented by child classes)
        self.setup_parameters()
        
        # Initialize fixed parameters now that param_dict is defined
        self._initialize_fixed_parameters()
        
        # Generate bounds
        self._generate_bounds()
    
    
    # =================================================================
    # Setup parameters (Must be implemented in Child Class)
    # =================================================================
    def setup_parameters(self):
        """
        Abstract method to be implemented by child classes.
        Should set self.n_states, self.total_parameters, and self.param_dict
        """
        raise NotImplementedError("Child classes must implement _setup_parameters")
    

    # =================================================================
    # Create Continuous Time State Space Model (Must be implemented in Child Class)
    # =================================================================
    def create_model(self, par):

        raise NotImplementedError("Subclasses must implement create_model method")
     
        
    # =================================================================
    # Scale Parameters
    # =================================================================  
    def scale_parameters(self, par):
        """
        Scale raw parameters based on their defined scale factors in param_dict.
        
        Args:
            par: List of raw parameter values
            
        Returns:
            Dictionary of scaled parameters with meaningful names
        """
        scaled_params = {}
        
        # Scale all parameters except for initial states
        for i in range(15):
            param_name = self.param_dict[i][0]
            scale_factor = self.param_dict[i][2]
            scaled_params[param_name] = par[i] * scale_factor
        
        # Handle initial states separately
        x0 = np.array([
            [self.iddata.y[0,0]],  # First state initialized with first measurement
            [par[15]],             # T2[0]
            [par[16]]              # T3[0] (implied from the model structure)
        ])
        
        scaled_params['x0'] = x0
        
        return scaled_params
    
    
    # =================================================================
    # Initialize Fixed Parameters
    # =================================================================
    def _initialize_fixed_parameters(self):
        """Initialize fixed parameters functionality."""
        # Create a mapping from parameter names to indices
        self.param_name_to_idx = {self.param_dict[i][0]: i for i in range(self.total_parameters)}
        
        # Determine which parameters are free and which are fixed
        self.free_indices = []
        self.fixed_indices = []
        self.fixed_values = {}
        
        # Use _fixed_params that was stored in __init__
        for param_name, value in self._fixed_params.items():
            if param_name in self.param_name_to_idx:
                idx = self.param_name_to_idx[param_name]
                self.fixed_indices.append(idx)
                # Store the physical value
                self.fixed_values[idx] = value
            else:
                raise ValueError(f"Unknown parameter name: {param_name}")
        
        # All parameters not in fixed_indices are free
        self.free_indices = [i for i in range(self.total_parameters) if i not in self.fixed_indices]
        
        # Update n_parameters based on free parameters
        self.n_parameters = len(self.free_indices)
    
    
    # =================================================================
    # Exand Parameters
    # =================================================================
    def _expand_parameters(self, reduced_par):
        """
        Expand the reduced parameter vector to include fixed parameters.
        
        Args:
            reduced_par: Array of free parameter values
            
        Returns:
            Array: Full parameter vector
        """
        full_par = np.zeros(self.total_parameters)
        
        # Fill in free parameters
        for i, free_idx in enumerate(self.free_indices):
            full_par[free_idx] = reduced_par[i]
        
        # Fill in fixed parameters
        for fixed_idx, value in self.fixed_values.items():
            # Convert physical value to scaled value
            scale_factor = self.param_dict[fixed_idx][2]
            full_par[fixed_idx] = value / scale_factor
        
        return full_par
    
    
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
    # Get Number of Parameters
    # =================================================================
    def get_n_parameters(self):
        
        if self.kstep>0:
            return self.n_parameters + self.n_states
        else:
            return self.n_parameters


    # =================================================================
    # Get Bounds
    # =================================================================
    def get_bounds(self):
         
        self._generate_bounds()
        
        if self.kstep>0:
            lb = np.append(self.lower_bounds,self.lower_bounds_feedback)
            ub = np.append(self.upper_bounds,self.upper_bounds_feedback) 
            return (lb, ub)
        else:
            return (self.lower_bounds, self.upper_bounds)

    
    # =================================================================
    # Get Bounds
    # =================================================================
    def _generate_bounds(self):
        self.lower_bounds = np.zeros(self.n_parameters)
        self.upper_bounds = np.full(self.n_parameters, np.inf)
        self.lower_bounds_feedback = np.zeros(self.n_states)
        self.upper_bounds_feedback = np.full(self.n_states, np.inf)


    # =================================================================
    # Print Parameters
    # =================================================================
    def print(self, scale=False):
        """Print all parameters with optional scaling and with confidence intervals if provided.
        
        Args:
            scale: If False, parameters are displayed with physical units.
                  If True, parameters are displayed without scaling (normalized).
        """
        if not hasattr(self, 'par') or self.par is None:
            print("No parameter estimates available.")
            return
            
        if not hasattr(self, 'param_dict'):
            # Fallback if param_dict is not defined
            for i, p in enumerate(self.par):
                print(f"Parameter {i}: {p:.3f}")
                if self.conf_intervals is not None and i < self.conf_intervals.shape[1]:
                    lower, upper = self.conf_intervals[0, i], self.conf_intervals[1, i]
                    print(f"    95% CI: [{lower:.3f}, {upper:.3f}]")
            return
        
        # Handle parameter expansion if the model supports fixed parameters
        full_par = self.par
        if hasattr(self, '_expand_parameters'):
            if self.kstep > 0 and len(self.par) > self.n_parameters:
                # For feedback case, we need to separate model parameters from feedback parameters
                model_params = self.par[:self.n_parameters]
                feedback_params = self.par[self.n_parameters:]
                # Expand only the model parameters
                expanded_model_params = self._expand_parameters(model_params)
                # Recombine with feedback parameters
                full_par = np.concatenate([expanded_model_params, feedback_params])
            elif len(self.par) == self.n_parameters:
                # Standard case without feedback
                full_par = self._expand_parameters(self.par)
            
        # Handle confidence interval expansion if needed
        full_conf = self.conf_intervals
        if hasattr(self, 'free_indices') and self.conf_intervals is not None:
            if self.kstep > 0 and len(self.par) > self.n_parameters:
                # For feedback case, handle model params and feedback params separately
                model_conf = self.conf_intervals[:, :self.n_parameters]
                feedback_conf = self.conf_intervals[:, self.n_parameters:]
                
                # Create full confidence intervals for model parameters
                expanded_model_conf = np.zeros((2, len(self.param_dict)))
                for i, free_idx in enumerate(self.free_indices):
                    if i < model_conf.shape[1]:
                        expanded_model_conf[0, free_idx] = model_conf[0, i]
                        expanded_model_conf[1, free_idx] = model_conf[1, i]
                
                # Combine expanded model confidence intervals with feedback confidence intervals
                full_conf = np.zeros((2, len(self.param_dict) + feedback_conf.shape[1]))
                full_conf[:, :len(self.param_dict)] = expanded_model_conf
                full_conf[:, len(self.param_dict):] = feedback_conf
            else:
                # Standard case without feedback
                full_conf = np.zeros((2, len(self.param_dict)))
                for i, free_idx in enumerate(self.free_indices):
                    if i < self.conf_intervals.shape[1]:
                        full_conf[0, free_idx] = self.conf_intervals[0, i]
                        full_conf[1, free_idx] = self.conf_intervals[1, i]
            
        # Print model parameters
        for idx, param_info in self.param_dict.items():
            if idx < len(full_par) and (idx < self.total_parameters):  # Only print model parameters here
                name, unit = param_info[0], param_info[1]
                scale_factor = param_info[2]                
                
                if not scale:                    
                    display_scale_factor = 1
                    value = full_par[idx]
                else:
                    unit = "-"
                    display_scale_factor = scale_factor
                    value = full_par[idx] * display_scale_factor
                    
                
                # Check if this is a fixed parameter
                is_fixed = hasattr(self, 'fixed_indices') and idx in self.fixed_indices
                
                if is_fixed:
                    print(f"{name}: {value:.3f} {unit} (FIXED)")
                else:
                    print(f"{name}: {value:.3f} {unit}")
                    
                    # Print confidence interval if available
                    if full_conf is not None and idx < full_conf.shape[1]:
                        lower = full_conf[0, idx] * display_scale_factor
                        upper = full_conf[1, idx] * display_scale_factor
                        print(f"    95% CI: [{lower:.3f}, {upper:.3f}] {unit}")
        
        # Print feedback parameters if present
        if self.kstep > 0 and len(full_par) > self.total_parameters:
            print("\nFeedback Parameters:")
            for i in range(self.total_parameters, len(full_par)):
                state_idx = i - self.total_parameters
                print(f"K_{state_idx+1}: {full_par[i]:.3f}")
                if full_conf is not None and i < full_conf.shape[1]:
                    lower = full_conf[0, i]
                    upper = full_conf[1, i]
                    print(f"    95% CI: [{lower:.3f}, {upper:.3f}]")
    
    
    # =================================================================
    # Plot Parameters and Confidence Intervals
    # ================================================================= 
    def plot_parameters(self, scale=False, figsize=(10, 8)):
        """Plot parameters with their confidence intervals.
        
        Args:
            figsize: Figure size as tuple (width, height)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if self.par is None or not any(self.par):
            print("Missing Estimated Parameter(s)")
            return
        
        # Handle parameter expansion if the model supports fixed parameters
        full_par = self.par
        if hasattr(self, '_expand_parameters'):
            if self.kstep > 0 and len(self.par) > self.n_parameters:
                # For feedback case, we need to separate model parameters from feedback parameters
                model_params = self.par[:self.n_parameters]
                feedback_params = self.par[self.n_parameters:]
                # Expand only the model parameters
                expanded_model_params = self._expand_parameters(model_params)
                # Recombine with feedback parameters
                full_par = np.concatenate([expanded_model_params, feedback_params])
            elif len(self.par) == self.n_parameters:
                # Standard case without feedback
                full_par = self._expand_parameters(self.par)
        
        if not hasattr(self, 'param_dict'):
            # Fallback if param_dict is not defined
            param_names = [f"Parameter {i}" for i in range(len(full_par))]
            param_values = full_par
            param_indices = list(range(len(full_par)))
            is_fixed = [False] * len(full_par)
        else:
            # Extract names from param_dict and handle feedback parameters
            param_names = []
            param_indices = []  # Store original indices
            units = []
            is_fixed = []  # Track which parameters are fixed
            
            # First, add model parameters
            for idx in sorted(self.param_dict.keys()):
                if idx < self.total_parameters:  # Only include model parameters here
                    name, unit = self.param_dict[idx][0], self.param_dict[idx][1]
                    param_names.append(name)
                    param_indices.append(idx)  # Store original index
                    units.append(unit)
                    
                    # Check if this parameter is fixed
                    fixed = hasattr(self, 'fixed_indices') and idx in self.fixed_indices
                    is_fixed.append(fixed)
            
            # Include feedback parameters if present
            if self.kstep > 0 and len(full_par) > self.total_parameters:
                for i in range(self.n_states):
                    param_names.append(f"K_{i+1}")
                    param_indices.append(self.total_parameters + i)  # Store original index
                    units.append("")
                    is_fixed.append(False)  # Feedback parameters are never fixed
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(param_names))
        
        # Calculate scaled parameter values
        param_values = []
        for i, idx in enumerate(param_indices):
            if idx < self.total_parameters and hasattr(self, 'param_dict'):
                # Apply scaling based on user preference
                if not scale:
                    value = full_par[idx]  # No rescaling
                else:
                    value = full_par[idx] * self.param_dict[idx][2]  # Apply scaling
            else:
                # Feedback parameters or fallback
                value = full_par[idx]
            param_values.append(value)
        
        # Plot the parameter values with different colors for fixed vs free
        bars = []
        for i, (val, fixed) in enumerate(zip(param_values, is_fixed)):
            color = 'lightgray' if fixed else 'skyblue'
            bar = ax.bar(i, val, width=0.5, color=color, alpha=0.7)
            bars.extend(bar)
        
        # Handle confidence interval expansion if needed
        full_conf = self.conf_intervals
        if hasattr(self, 'free_indices') and self.conf_intervals is not None:
            if self.kstep > 0 and len(self.par) > self.n_parameters:
                # For feedback case, handle model params and feedback params separately
                model_conf = self.conf_intervals[:, :self.n_parameters]
                feedback_conf = self.conf_intervals[:, self.n_parameters:]
                
                # Create full confidence intervals for model parameters
                expanded_model_conf = np.zeros((2, len(self.param_dict)))
                for i, free_idx in enumerate(self.free_indices):
                    if i < model_conf.shape[1]:
                        expanded_model_conf[0, free_idx] = model_conf[0, i]
                        expanded_model_conf[1, free_idx] = model_conf[1, i]
                
                # Combine expanded model confidence intervals with feedback confidence intervals
                total_params = len(self.param_dict) + self.n_states
                full_conf = np.zeros((2, total_params))
                full_conf[:, :len(self.param_dict)] = expanded_model_conf
                full_conf[:, len(self.param_dict):] = feedback_conf
            else:
                # Standard case without feedback
                full_conf = np.zeros((2, len(self.param_dict)))
                for i, free_idx in enumerate(self.free_indices):
                    if i < self.conf_intervals.shape[1]:
                        full_conf[0, free_idx] = self.conf_intervals[0, i]
                        full_conf[1, free_idx] = self.conf_intervals[1, i]
        
        # Plot confidence intervals if provided (only for non-fixed params)
        if full_conf is not None and full_conf.shape[1] > 0:
            for i, (original_idx, fixed) in enumerate(zip(param_indices, is_fixed)):
                if not fixed and i < len(param_indices) and i < full_conf.shape[1]:
                    # Get confidence intervals and apply scaling if needed
                    if original_idx < self.total_parameters and hasattr(self, 'param_dict'):
                        if not scale:
                            lower = full_conf[0, i]
                            upper = full_conf[1, i]
                        else:
                            lower = full_conf[0, i] * self.param_dict[original_idx][2]
                            upper = full_conf[1, i] * self.param_dict[original_idx][2]
                    else:
                        # Feedback parameters
                        lower = full_conf[0, i]
                        upper = full_conf[1, i]
                    
                    # Draw error bars
                    ax.plot([i, i], [lower, upper], 'r-', linewidth=2)
                    ax.plot([i-0.1, i+0.1], [lower, lower], 'r-', linewidth=2)
                    ax.plot([i-0.1, i+0.1], [upper, upper], 'r-', linewidth=2)
        
        # Add value labels on the bars
        for bar, value, fixed in zip(bars, param_values, is_fixed):
            height = bar.get_height()
            if fixed:
                label_text = f'{value:.2f} (FIXED)'
            else:
                label_text = f'{value:.2f}'
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05 * max(abs(np.array(param_values))),
                   label_text, ha='center', va='bottom', fontsize=9, rotation=45)
        
        # Set axis labels and title
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Value')
        ax.set_title('Parameter Estimates with 95% Confidence Intervals')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        
        # Add units to the parameter names in the legend if available
        if 'units' in locals() and len(units) == len(param_names):
            handles = [
                plt.Rectangle((0,0),1,1, color='skyblue', alpha=0.7),
                plt.Rectangle((0,0),1,1, color='lightgray', alpha=0.7)
            ]
            labels = ['Free Parameter', 'Fixed Parameter']
            
            # Add a divider in the legend if we have feedback parameters
            if self.kstep > 0 and len(full_par) > self.total_parameters:
                handles.append(plt.Rectangle((0,0),1,1, color='white', alpha=0))  # Invisible rectangle as spacer
                labels.append('')
                handles.append(plt.Rectangle((0,0),1,1, color='skyblue', alpha=0.7))
                labels.append('Feedback Parameter')
            
            ax.legend(handles, labels, loc='upper right')
            
            # Create a secondary y-axis for the units
            ax2 = ax.twinx()
            ax2.set_yticks([])
            
            # Add unit text annotations
            for i, (name, unit) in enumerate(zip(param_names, units)):
                if unit:  # Only add if unit is not empty
                    ax.annotate(f"[{unit}]", xy=(i, 0), xytext=(i, -0.05 * max(abs(np.array(param_values)))), 
                                ha='center', va='top', fontsize=8)
        
        plt.tight_layout()
        return fig