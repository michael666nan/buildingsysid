import numpy as np
from buildingsysid.criterion_of_fit.base_greybox import BaseGreyBox
from buildingsysid.criterion_of_fit.statespace import StateSpace


# =================================================================
# FIRST ORDER MODEL
# =================================================================
class First(BaseGreyBox):
    
    def setup_parameters(self):
        """Set up parameters for the first-order model"""
        
        # Define states and maximum parameters
        self.n_states = 1
        self.total_parameters = 4
        
        # Define parameter dictionary
        self.param_dict = {
            0: ("H_1a", "W/(K*m²)", self.floor_area),
            1: ("C_1", "Wh/(K*m²)", self.floor_area * 3600),
            2: ("ws1", "m²", 1.0),
            3: ("wh1", "-", 1.0)
        }
    
    
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
        
        # Extract scaled parameters
        H_1a = full_par[0] * self.param_dict[0][2]  # (W/K)
        C_1 = full_par[1] * self.param_dict[1][2]   # (J/K)
        ws1 = full_par[2] * self.param_dict[2][2]   # (m²)
        wh1 = full_par[3] * self.param_dict[3][2]   # (-)
        
        # Initial state vector
        x0 = np.array([[self.iddata.y[0,0]]])
        
        # Calculate the A matrix
        A = np.array([[-H_1a / C_1]])
        
        # Construct the B matrix (inputs: ambient temp, solar radiation, heating)
        B = np.array([[H_1a / C_1, ws1 / C_1, wh1 / C_1]])
        
        # C matrix (output is the state temperature)
        C = np.array([[1]])
        
        # D matrix (no direct feedthrough)
        D = np.array([[0, 0, 0]])
        
        # For feedback matrix
        K = self.feedback_matrix(reduced_par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)
    
    
# =================================================================
# Second-order model
# =================================================================
class Second(BaseGreyBox):
    
    def setup_parameters(self):
        """Set up parameters for the second-order model"""
        
        # Define states and maximum parameters
        self.n_states = 2
        self.total_parameters = 10
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("H_12", "W/(K*m²)", self.floor_area),
            1: ("H_1a", "W/(K*m²)", self.floor_area),
            2: ("H_2a", "W/(K*m²)", self.floor_area),
            3: ("C_1", "Wh/(K*m²)", self.floor_area * 3600),
            4: ("C_2", "Wh/(K*m²)", self.floor_area * 3600),
            5: ("ws1", "m²", 1.0),
            6: ("ws2", "m²", 1.0),
            7: ("wh1", "-", 1.0),
            8: ("wh2", "-", 1.0),
            9: ("T2[0]", "°C", 1.0),
        }
    
    def create_model(self, reduced_par):
        """Create the second-order state-space model"""
        # Expand parameters if needed
        full_par = self._expand_parameters(reduced_par)
        
        # Extract scaled parameters
        H_12 = full_par[0] * self.param_dict[0][2]  # (W/K)
        H_1a = full_par[1] * self.param_dict[1][2]  # (W/K)
        H_2a = full_par[2] * self.param_dict[2][2]  # (W/K)
        C_1 = full_par[3] * self.param_dict[3][2]   # (J/K)
        C_2 = full_par[4] * self.param_dict[4][2]   # (J/K)
        ws1 = full_par[5] * self.param_dict[5][2]    # (m²)
        ws2 = full_par[6] * self.param_dict[6][2]    # (m²)
        wh1 = full_par[7] * self.param_dict[7][2]    # (-)
        wh2 = full_par[8] * self.param_dict[8][2]    # (-)
        
        # Calculate the elements of the A matrix
        a11 = -(H_1a + H_12) / C_1
        a12 = H_12 / C_1
        a21 = H_12 / C_2
        a22 = -(H_2a + H_12) / C_2
        
        # Construct the matrix A
        A = np.array([
            [a11, a12],
            [a21, a22]
        ])
        
        # Construct the B matrix (inputs: ambient temp, solar radiation, heating)
        B = np.array([
            [H_1a / C_1, ws1 / C_1, wh1 / C_1],
            [H_2a / C_2, ws2 / C_2, wh2 / C_2]
        ])
        
        # C matrix (output is the first state temperature)
        C = np.array([[1, 0]])
        
        # D matrix (no direct feedthrough)
        D = np.array([[0, 0, 0]])
        
        # Initial state vector - initialize first state from measurement, second state as average
        x0 = np.array([
            [self.iddata.y[0,0]],   # First state from first measurement
            [full_par[9]]           # Second state is parameterized)
        ])
        
        # For feedback matrix
        K = self.feedback_matrix(reduced_par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)
    
    
# =================================================================
# Third-order model
# =================================================================    
class Third(BaseGreyBox):
    
    def setup_parameters(self):
        """Set up parameters for the second-order model"""
        
        # Define states and maximum parameters
        self.n_states = 3
        self.total_parameters = 17
        
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
            15: ("T2[0]", "°C", 1.0),
            16: ("T3[0]", "°C", 1.0)
        }
    
    def create_model(self, reduced_par):
        """Create the second-order state-space model"""
        # Expand parameters if needed
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
        
        K = self.feedback_matrix(reduced_par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)
    
    
# =================================================================
# Fourth-order model
# =================================================================    
class Fourth(BaseGreyBox):
    
    def setup_parameters(self):
        """Set up parameters for the fourth-order model"""
        
        # Define states and maximum parameters
        self.n_states = 4
        self.total_parameters = 25
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("H_12", "W/(K*m²)", self.floor_area),
            1: ("H_13", "W/(K*m²)", self.floor_area),
            2: ("H_14", "W/(K*m²)", self.floor_area),
            3: ("H_23", "W/(K*m²)", self.floor_area),
            4: ("H_24", "W/(K*m²)", self.floor_area),
            5: ("H_34", "W/(K*m²)", self.floor_area),
            6: ("H_1a", "W/(K*m²)", self.floor_area),
            7: ("H_2a", "W/(K*m²)", self.floor_area),
            8: ("H_3a", "W/(K*m²)", self.floor_area),
            9: ("H_4a", "W/(K*m²)", self.floor_area),
            10: ("C_1", "Wh/(K*m²)", self.floor_area * 3600),
            11: ("C_2", "Wh/(K*m²)", self.floor_area * 3600),
            12: ("C_3", "Wh/(K*m²)", self.floor_area * 3600),
            13: ("C_4", "Wh/(K*m²)", self.floor_area * 3600),
            14: ("ws1", "m²", 1.0),
            15: ("ws2", "m²", 1.0),
            16: ("ws3", "m²", 1.0),
            17: ("ws4", "m²", 1.0),
            18: ("wh1", "-", 1.0),
            19: ("wh2", "-", 1.0),
            20: ("wh3", "-", 1.0),
            21: ("wh4", "-", 1.0),
            22: ("T2[0]", "°C", 1.0),
            23: ("T3[0]", "°C", 1.0),
            24: ("T4[0]", "°C", 1.0)
        }
    
    def create_model(self, reduced_par):
        """Create the fourth-order state-space model"""
        # Expand parameters if needed
        full_par = self._expand_parameters(reduced_par)
        
        # Extract scaled parameters
        H_12 = full_par[0] * self.param_dict[0][2]  # (W/K)
        H_13 = full_par[1] * self.param_dict[1][2]  # (W/K)
        H_14 = full_par[2] * self.param_dict[2][2]  # (W/K)
        H_23 = full_par[3] * self.param_dict[3][2]  # (W/K)
        H_24 = full_par[4] * self.param_dict[4][2]  # (W/K)
        H_34 = full_par[5] * self.param_dict[5][2]  # (W/K)
        H_1a = full_par[6] * self.param_dict[6][2]  # (W/K)
        H_2a = full_par[7] * self.param_dict[7][2]  # (W/K)
        H_3a = full_par[8] * self.param_dict[8][2]  # (W/K)
        H_4a = full_par[9] * self.param_dict[9][2]  # (W/K)
        C_1 = full_par[10] * self.param_dict[10][2]  # (J/K)
        C_2 = full_par[11] * self.param_dict[11][2]  # (J/K)
        C_3 = full_par[12] * self.param_dict[12][2]  # (J/K)
        C_4 = full_par[13] * self.param_dict[13][2]  # (J/K)
        ws1 = full_par[14] * self.param_dict[14][2]  # (m²)
        ws2 = full_par[15] * self.param_dict[15][2]  # (m²)
        ws3 = full_par[16] * self.param_dict[16][2]  # (m²)
        ws4 = full_par[17] * self.param_dict[17][2]  # (m²)
        wh1 = full_par[18] * self.param_dict[18][2]  # (-)
        wh2 = full_par[19] * self.param_dict[19][2]  # (-)
        wh3 = full_par[20] * self.param_dict[20][2]  # (-)
        wh4 = full_par[21] * self.param_dict[21][2]  # (-)
        
        # Initial state vector - first state from measurement, others from parameters
        x0 = np.array([
            [self.iddata.y[0,0]],  # First state from first measurement
            [full_par[22]],        # T2[0]
            [full_par[23]],        # T3[0]
            [full_par[24]]         # T4[0]
        ])
        
        # Calculate the elements of the A matrix
        a11 = -(H_1a + H_12 + H_13 + H_14) / C_1
        a12 = H_12 / C_1
        a13 = H_13 / C_1
        a14 = H_14 / C_1
        
        a21 = H_12 / C_2
        a22 = -(H_2a + H_12 + H_23 + H_24) / C_2
        a23 = H_23 / C_2
        a24 = H_24 / C_2
        
        a31 = H_13 / C_3
        a32 = H_23 / C_3
        a33 = -(H_3a + H_13 + H_23 + H_34) / C_3
        a34 = H_34 / C_3
        
        a41 = H_14 / C_4
        a42 = H_24 / C_4
        a43 = H_34 / C_4
        a44 = -(H_4a + H_14 + H_24 + H_34) / C_4
        
        # Construct the A matrix
        A = np.array([
            [a11, a12, a13, a14],
            [a21, a22, a23, a24],
            [a31, a32, a33, a34],
            [a41, a42, a43, a44]
        ])
        
        # Construct the B matrix (inputs: ambient temp, solar radiation, heating)
        B = np.array([
            [H_1a / C_1, ws1 / C_1, wh1 / C_1],
            [H_2a / C_2, ws2 / C_2, wh2 / C_2],
            [H_3a / C_3, ws3 / C_3, wh3 / C_3],
            [H_4a / C_4, ws4 / C_4, wh4 / C_4]
        ])
        
        # C matrix (output is the first state temperature)
        C = np.array([[1, 0, 0, 0]])
        
        # D matrix (no direct feedthrough)
        D = np.array([[0, 0, 0]])
        
        # For feedback matrix
        K = self.feedback_matrix(reduced_par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)