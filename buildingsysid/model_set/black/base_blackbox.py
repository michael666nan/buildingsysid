import numpy as np

class BaseBlackBox:
    """
    Base class for black-box models used in system identification.
    """
    
    def __init__(self):
        """
        Initialize the black-box model.
        """
        
        # Number of Parameters
        self.n_free_parameters = None  # Number of free parameters
        self.n_states = None
        self.total_parameters = None  # Total number of parameters (free + fixed)
        
        # Parameter fixing functionality
        self._fixed_params = {}
        self.free_indices = []
        self.fixed_indices = []
        self.fixed_values = {}
        self.param_name_to_idx = {}
        
        # Parameter bounds functionality
        self._param_bounds = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        
        # Estimation Results
        self.result = None

    
    # =================================================================
    # Create Discrete Time State Space Model (Must be implemented in Child Class)
    # =================================================================
    def create_model(self, par):
        """
        Create a state-space model from the parameters.
        
        This method must be implemented by child classes.
        
        Args:
            par: Parameter vector
            
        Returns:
            StateSpace: The created state-space model
        """
        raise NotImplementedError("Subclasses must implement create_model method")
    
    
    # =================================================================
    # Called when Child Class is Initialized
    # ================================================================= 
    def _initialize_fixed_parameters(self, fixed_params={}, param_bounds={}):
       """Initialize fixed parameters and bounds functionality."""
       
       self._fixed_params = fixed_params
       self._param_bounds = param_bounds
       
       # Create a mapping from parameter names to indices
       self.param_name_to_idx = {self.param_dict[i][0]: i for i in range(self.total_parameters)}
       
       # Determine which parameters are free and which are fixed
       self.free_indices = []
       self.fixed_indices = []
       self.fixed_values = {}
       
       # Initialize bounds dictionaries
       self.lower_bounds = {}
       self.upper_bounds = {}
       
       # Process fixed parameters from initialization
       for param_name, value in self._fixed_params.items():
           if param_name in self.param_name_to_idx:
               idx = self.param_name_to_idx[param_name]
               self.fixed_indices.append(idx)
               self.fixed_values[idx] = value
           else:
               raise ValueError(f"Unknown parameter name: {param_name}")
       
       # Process parameter bounds from initialization
       for param_name, bounds in self._param_bounds.items():
           if param_name in self.param_name_to_idx:
               idx = self.param_name_to_idx[param_name]
               if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                   lower, upper = bounds
                   self.lower_bounds[idx] = lower
                   self.upper_bounds[idx] = upper
               else:
                   raise ValueError(f"Bounds for {param_name} must be a list or tuple of [lower, upper]")
           else:
               raise ValueError(f"Unknown parameter name: {param_name}")
       
       # All parameters not in fixed_indices are free
       self.free_indices = [i for i in range(self.total_parameters) if i not in self.fixed_indices]
       
       # Update n_parameters based on free parameters
       self.n_free_parameters = len(self.free_indices)
    
    
    # =================================================================
    # Called by Optimization Class using Objective Function 
    # ================================================================= 
    def set_feedback_mode(self, use_feedback):
        """
        Set whether to use feedback in the model or not.
        
        Args:
            use_feedback: If True, use default feedback; if False, fix them all to zero
        """         
        if use_feedback: 
            return 
        else:
            # Fix all feedback parameters to zero
            for i in range(self.n_states):
                param_name = f"k{i+1}"
                idx = self.param_name_to_idx[param_name]
                if idx not in self.fixed_indices:
                    self.fixed_indices.append(idx)
                self.fixed_values[idx] = 0
                
        # All parameters not in fixed_indices are free
        self.free_indices = [i for i in range(self.total_parameters) if i not in self.fixed_indices]
        
        # Update n_parameters based on free parameters
        self.n_free_parameters = len(self.free_indices)
    
    
    # =================================================================
    # Called by Optimization Class to create Bounds
    # ================================================================= 
    def get_parameter_index(self, param_name):
        """
        Get the index of a parameter in the free parameter vector.
        
        Args:
            param_name: Name of the parameter
            
        Returns:
            Integer index in the free parameter vector, or None if not found
        """
        # Check if param_name exist
        if param_name not in self.param_name_to_idx:
            return None
        
        # Check if param_name is Free
        full_idx = self.param_name_to_idx[param_name]
        if full_idx in self.fixed_indices:
            return None
            
        # Find the position of this parameter in the free_indices list
        return self.free_indices.index(full_idx)
    

    # =================================================================
    # Called by Child Class to Create Full State Space Model
    # =================================================================    
    def _expand_parameters(self, par):
        
        # Expand the model parameters
        full_model_par = np.zeros(self.total_parameters)
        
        # Fill in free model parameters
        for i, free_idx in enumerate(self.free_indices):
            full_model_par[free_idx] = par[i]
        
        # Fill in fixed model parameters
        for fixed_idx, value in self.fixed_values.items():
            full_model_par[fixed_idx] = value
        
        return full_model_par

    
    # =================================================================
    # Called by Optimization Class to get Bounds
    # ================================================================= 
    def get_parameter_bounds(self):
        """
        Get the bounds for all free parameters.
        
        Returns:
            tuple: (lower_bounds, upper_bounds) for all free parameters
        """
        # Default bounds if none specified (can be adjusted as needed)
        default_lower = -np.inf
        default_upper = np.inf
        
        # Create bounds arrays for free parameters
        lower = np.full(self.n_free_parameters, default_lower)
        upper = np.full(self.n_free_parameters, default_upper)
        
        # Fill in specified bounds for free parameters
        for full_idx, lower_val in self.lower_bounds.items():
            if full_idx in self.free_indices:
                free_idx = self.free_indices.index(full_idx)
                lower[free_idx] = lower_val
                
        for full_idx, upper_val in self.upper_bounds.items():
            if full_idx in self.free_indices:
                free_idx = self.free_indices.index(full_idx)
                upper[free_idx] = upper_val
        
        return lower, upper
    
    
    # =================================================================
    # Get Estimated State Space Model
    # =================================================================
    def get_state_space(self):
        """
        Get the estimated state space model.
        
        Returns:
            State Space Model
        """
        if self.result is None:
            return None
            
                
        return self.create_model(self.result.x)
    

    # =================================================================
    # Print Parameters
    # =================================================================
    def print(self):
        """Print all parameters with confidence intervals and bounds if provided."""
        
        if self.result is None:
            print("No parameter estimates available.")
            return
        
        par = self.result["x"] 
        conf_intervals = self.result.get("confidence_intervals")
        
        if not hasattr(self, 'param_dict'):
            # Fallback if param_dict is not defined
            for i, p in enumerate(par):
                print(f"Parameter {i}: {p:.3f}")
                if conf_intervals is not None and i < conf_intervals.shape[1]:
                    lower, upper = conf_intervals[0, i], conf_intervals[1, i]
                    print(f"    95% CI: [{lower:.3f}, {upper:.3f}]")
            return
        
        # Expand parameters to get full vectors
        full_model_par = self._expand_parameters(par)
        
        # Handle confidence interval expansion if needed
        full_model_conf = None
        
        if conf_intervals is not None:
            full_model_conf = np.zeros((2, self.total_parameters))
            for i, free_idx in enumerate(self.free_indices):
                if i < conf_intervals.shape[1]:
                    full_model_conf[0, free_idx] = conf_intervals[0, i]
                    full_model_conf[1, free_idx] = conf_intervals[1, i]
        
        # Get all bounds using the existing method
        all_lower_bounds, all_upper_bounds = self.get_parameter_bounds()
        
        # Print model parameters
        for idx, param_info in self.param_dict.items():
            name, unit = param_info[0], param_info[1]
            value = full_model_par[idx]
                
            # Check if this is a fixed parameter
            is_fixed = hasattr(self, 'fixed_indices') and idx in self.fixed_indices
            
            if is_fixed:
                print(f"{name}: {value:.3f} {unit} (FIXED)")
            else:
                print(f"{name}: {value:.3f} {unit}")
                
            # Print confidence interval if available
            if full_model_conf is not None and idx < full_model_conf.shape[1]:
                lower = full_model_conf[0, idx]
                upper = full_model_conf[1, idx]
                print(f"    95% CI: [{lower:.3f}, {upper:.3f}] {unit}")
            
            # Only print bounds if the parameter is not fixed
            if not is_fixed:
                # Find this parameter's position in the free_indices list to get its bounds
                if idx in self.free_indices:
                    free_idx = self.free_indices.index(idx)
                    lower = all_lower_bounds[free_idx]
                    upper = all_upper_bounds[free_idx]
                    
                    # Format the bounds nicely
                    lower_str = f"{lower:.3f}" if np.isfinite(lower) else "-∞"
                    upper_str = f"{upper:.3f}" if np.isfinite(upper) else "∞"
                    
                    print(f"    Bounds: [{lower_str}, {upper_str}] {unit}")