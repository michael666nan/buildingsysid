import numpy as np

from criterion_of_fit.base_greybox import BaseGreyBox
from criterion_of_fit.statespace import StateSpace


# =================================================================
# Full Model - NOT IDENTIFIABLE
# =================================================================
class Full(BaseGreyBox):
    
    def __init__(self, floor_area):
        super().__init__(floor_area)
        self.n_parameters = 4
        self.n_states = 1
        
        self._generate_bounds()
    
    
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("H_a1", "W/(K*m²)", self.floor_area),
            1: ("C_1", "Wh/(K*m²)", self.floor_area * 3600),
            2: ("gA", "(m²/m²)", self.floor_area),
            3: ("wh", "-", 1.0),
        }
    
    
    def create_model(self, par):
        # Use the scales defined in param_dict
        H_a1 = par[0] * self.floor_area                # (W/K)
        C_1 = par[1] * self.floor_area * 3600          # (J/K)
        gA = par[2] * self.floor_area                  # (m²)
        wh = par[3]
        
        A = np.array([[-H_a1/C_1]])
        B = np.array([[H_a1/C_1, gA/C_1, wh/C_1]])
        C = np.array([[1]])
        D = np.array([[0, 0, 0]])
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]]])
        
        K = self.feedback_matrix(par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)
    



# =================================================================
# Convector - Indentifiable 
# Assumptions:  - 100 % convection
# =================================================================  
class Convector(BaseGreyBox):
    
    def __init__(self, floor_area):
        super().__init__(floor_area)
        self.n_parameters = 3
        self.n_states = 1
        
        self._generate_bounds()
    
    
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("H_a1", "W/(K*m²)", self.floor_area),
            1: ("C_1", "Wh/(K*m²)", self.floor_area * 3600),
            2: ("gA", "(m²/m²)", self.floor_area),
        }
    
    
    def create_model(self, par):
        # Use the scales defined in param_dict
        H_a1 = par[0] * self.floor_area                # (W/K)
        C_1 = par[1] * self.floor_area * 3600          # (J/K)
        gA = par[2] * self.floor_area                  # (m²)
        
        A = np.array([[-H_a1/C_1]])
        B = np.array([[H_a1/C_1, gA/C_1, 1/C_1]])
        C = np.array([[1]])
        D = np.array([[0, 0, 0]])
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]]])
        
        K = self.feedback_matrix(par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)
    
    
    
