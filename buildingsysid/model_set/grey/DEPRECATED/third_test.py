import numpy as np
from criterion_of_fit.base_greybox import BaseGreyBox
from criterion_of_fit.statespace import StateSpace

class Full(BaseGreyBox):
    
    def __init__(self, floor_area, fixed_params=None):
        # Initialize base class first
        super().__init__(floor_area)
        
        # Store fixed parameters
        self.fixed_params = fixed_params or {}
        
        # Define states and maximum parameters
        self.n_states = 3
        self.total_parameters = 17  # Total number of parameters in the full model
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("H_12", "W/(K*m²)", self.floor_area),
            1: ("H_13", "W/(K*m²)", self.floor_area),
            2: ("H_23", "W/(K*m²)", self.floor_area),
            3: ("H_1a", "W/(K*m²)", self.floor_area),
            4: ("H_2a", "W/(K*m²)", self.floor_area),
            5: ("H_3a", "W/(K*m²)", self.floor_area),
            6: ("C_1", "Wh/(K*m²)", self.floor_area * 3600),
            7: ("C_2", "Wh/(K*m²)", self.floor_area * 3600),
            8: ("C_3", "Wh/(K*m²)", self.floor_area * 3600),
            9: ("ws1", "m²", 1.0),
            10: ("ws2", "m²", 1.0),
            11: ("ws3", "m²", 1.0),
            12: ("wh1", "-", 1.0),
            13: ("wh2", "-", 1.0),
            14: ("wh3", "-", 1.0),
            15: ("T2[0]", "", 1.0),
            16: ("T3[0]", "", 1.0)
        }
        
        # Setup parameter mapping
        self._setup_parameter_mapping()
        
        # Set the number of parameters to be estimated
        self.n_parameters = len(self.free_indices)
        
        # Generate bounds for the free parameters
        self._generate_bounds()
    
    def _setup_parameter_mapping(self):
        """Set up mappings between free and fixed parameters"""
        # Create a mapping from parameter names to indices
        self.param_name_to_idx = {self.param_dict[i][0]: i for i in range(self.total_parameters)}
        
        # Determine which parameters are free and which are fixed
        self.free_indices = []
        self.fixed_indices = []
        self.fixed_values = {}
        
        for param_name, value in self.fixed_params.items():
            if param_name in self.param_name_to_idx:
                idx = self.param_name_to_idx[param_name]
                self.fixed_indices.append(idx)
                # Store the physical value
                self.fixed_values[idx] = value
            else:
                raise ValueError(f"Unknown parameter name: {param_name}")
        
        # All parameters not in fixed_indices are free
        self.free_indices = [i for i in range(self.total_parameters) if i not in self.fixed_indices]
    
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
    
    def create_model(self, reduced_par):
        """
        Create the state-space model with the given parameters.
        
        Args:
            reduced_par: Array of free parameter values
            
        Returns:
            StateSpace: State-space model
        """
        # Expand the reduced parameter vector to the full parameter vector
        full_par = self._expand_parameters(reduced_par)
        
        # Use the scales defined in param_dict
        H_12 = full_par[0] * self.param_dict[0][2]  # (W/K)
        H_13 = full_par[1] * self.param_dict[1][2]  # (W/K)
        H_23 = full_par[2] * self.param_dict[2][2]  # (W/K)
        H_1a = full_par[3] * self.param_dict[3][2]  # (W/K)
        H_2a = full_par[4] * self.param_dict[4][2]  # (W/K)
        H_3a = full_par[5] * self.param_dict[5][2]  # (W/K)
        C_1 = full_par[6] * self.param_dict[6][2]   # (J/K)
        C_2 = full_par[7] * self.param_dict[7][2]   # (J/K)
        C_3 = full_par[8] * self.param_dict[8][2]   # (J/K)
        ws1 = full_par[9] * self.param_dict[9][2]   # (m²)
        ws2 = full_par[10] * self.param_dict[10][2]   # (m²) 
        ws3 = full_par[11] * self.param_dict[11][2]   # (m²) 
        wh1 = full_par[12] * self.param_dict[12][2]   # (-)
        wh2 = full_par[13] * self.param_dict[13][2]   # (-)
        wh3 = full_par[14] * self.param_dict[14][2]   # (-)
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                      [full_par[15]],
                      [full_par[16]]])
        
        # Calculate the elements of the matrix
        a11 = -(H_1a + H_12 + H_13) / C_1
        a12 = H_12 / C_1
        a13 = H_13 / C_1
        
        a21 = H_12 / C_2
        a22 = -(H_2a + H_12 + H_23) / C_2
        a23 = H_23 / C_2
        
        a31 = H_13 / C_3
        a32 = H_23 / C_3
        a33 = -(H_3a + H_13 + H_23) / C_3
        
        # Construct the matrix A
        A = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ])
                
        B = np.array([
            [H_1a / C_1, ws1 / C_1, wh1 / C_1],
            [H_2a / C_2, ws2 / C_2, wh2 / C_2],
            [H_3a / C_3, ws3 / C_3, wh3 / C_3]
        ])
        
        C = np.array([[1, 0, 0]])
        D = np.array([[0, 0, 0]])
        
        # For feedback matrix, we need to use the original parameter vector
        # as it might contain the feedback coefficients after the model parameters
        K = self.feedback_matrix(reduced_par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)
    
    def print(self, scale=False):
        """
        Print the parameter estimates with confidence intervals.
        Override to handle fixed parameters.
        """
        if self.par is None:
            print("No parameter estimates available.")
            return
        
        # If we have a reduced parameter vector, expand it
        if len(self.par) == self.n_parameters:
            full_par = self._expand_parameters(self.par)
        else:
            full_par = self.par  # Assume it's already expanded
        
        # Expand confidence intervals if available
        full_conf = None
        if self.conf_intervals is not None:
            full_conf = np.zeros((2, self.total_parameters))
            for i, free_idx in enumerate(self.free_indices):
                if i < self.conf_intervals.shape[1]:
                    full_conf[0, free_idx] = self.conf_intervals[0, i]
                    full_conf[1, free_idx] = self.conf_intervals[1, i]
        
        # Print parameters
        for idx in range(self.total_parameters):
            name = self.param_dict[idx][0]
            unit = self.param_dict[idx][1]
            scale_factor = self.param_dict[idx][2]
            
            if not scale:
                print("No scaling")
                scale=1
            
            # Calculate physical value
            value = full_par[idx] * scale
            
            # Print with FIXED marker if it's a fixed parameter
            if idx in self.fixed_indices:
                print(f"{name}: {value:.2f} {unit} (FIXED)")
            else:
                print(f"{name}: {value:.2f} {unit}")
                
                # Print confidence interval if available
                if full_conf is not None:
                    lower = full_conf[0, idx] * scale
                    upper = full_conf[1, idx] * scale
                    print(f"    95% CI: [{lower:.2f}, {upper:.2f}] {unit}")
        
        # Print feedback parameters if present
        if self.kstep > 0 and len(self.par) > self.n_parameters:
            for i in range(self.n_parameters, len(self.par)):
                print(f"k{i - self.n_parameters + 1}: {self.par[i]:.2f}")
                if self.conf_intervals is not None and i < self.conf_intervals.shape[1]:
                    lower = self.conf_intervals[0, i]
                    upper = self.conf_intervals[1, i]
                    print(f"    95% CI: [{lower:.2f}, {upper:.2f}]")