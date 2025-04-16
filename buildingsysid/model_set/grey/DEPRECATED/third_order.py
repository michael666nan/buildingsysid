import numpy as np

from criterion_of_fit.base_greybox import BaseGreyBox
from criterion_of_fit.statespace import StateSpace


# =================================================================
# Full Model - NOT IDENTIFIABLE
# =================================================================
class Full(BaseGreyBox):
    
    def __init__(self, floor_area):
        super().__init__(floor_area)
        self.n_parameters = 17
        self.n_states = 3
        
        self._generate_bounds()    
    
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
            9: ("ws1", "(m²)", 1.0),
            10: ("ws2", "m²", 1.0),
            11: ("ws3", "m²", 1.0),
            12: ("wh1", "-", 1.0),
            13: ("wh2", "-", 1.0),
            14: ("wh3", "-", 1.0),
            15: ("T2[0]", "", 1.0),
            16: ("T3[0]", "", 1.0)
        }
    
    def create_model(self, par):
        # Get scaled parameters
        p = self.scale_parameters(par)
        
        # Calculate the elements of the matrix
        a11 = -(p['H_1a'] + p['H_12'] + p['H_13']) / p['C_1']
        a12 = p['H_12'] / p['C_1']
        a13 = p['H_13'] / p['C_1']
        
        a21 = p['H_12'] / p['C_2']
        a22 = -(p['H_2a'] + p['H_12'] + p['H_23']) / p['C_2']
        a23 = p['H_23'] / p['C_2']
        
        a31 = p['H_13'] / p['C_3']
        a32 = p['H_23'] / p['C_3']
        a33 = -(p['H_3a'] + p['H_13'] + p['H_23']) / p['C_3']
        
        # Construct the matrix A
        A = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ])
                
        B = np.array([
            [p['H_1a'] / p['C_1'], p['ws1'] / p['C_1'], p['wh1'] / p['C_1']],
            [p['H_2a'] / p['C_2'], p['ws2'] / p['C_2'], p['wh2'] / p['C_2']],
            [p['H_3a'] / p['C_3'], p['ws3'] / p['C_3'], p['wh3'] / p['C_3']]
        ])
        
        C = np.array([[1, 0, 0]])
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)
            
        return StateSpace(A, B, C, D, K, p['x0'], samplingTime=0)