import numpy as np

from system_identification.black.base_blackbox import BaseBlackBox
from system_identification.statespace import StateSpace



# =================================================================
# Full Model - NOT IDENTIFIABLE
# =================================================================
class Full(BaseBlackBox):

    def __init__(self):
        super().__init__()
        self.n_parameters = 20
        self.n_states = 3
        
        self._generate_bounds()
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("a11", "", 1.0),
            1: ("a12", "", 1.0),
            2: ("a13", "", 1.0),
            3: ("a21", "", 1.0),
            4: ("a22", "", 1.0),
            5: ("a23", "", 1.0),
            6: ("a31", "", 1.0),
            7: ("a32", "", 1.0),
            8: ("a33", "", 1.0),            
            9: ("b11", "", 1.0),
            10: ("b12", "", 1.0),
            11: ("b13", "", 1.0),
            12: ("b21", "", 1.0),
            13: ("b22", "", 1.0),
            14: ("b23", "", 1.0),
            15: ("b31", "", 1.0),
            16: ("b32", "", 1.0),
            17: ("b33", "", 1.0),
            18: ("x2[0]", "Celsius", 1.0),
            19: ("x3[0]", "Celsius", 1.0)
        }
    
    
    def create_model(self, par):
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[18]],
                       [par[19]]])
        
        # Discrete-time state-space matrices
        A = np.array([
            [par[0], par[1], par[2]],
            [par[3], par[4], par[5]],
            [par[6], par[7], par[8]]
        ])
        
        B = np.array([
            [par[9], par[10], par[11]],
            [par[12], par[13], par[14]],
            [par[15], par[16], par[17]]
            
        ])
        
        C = np.array([[1, 0, 0]])  # Only indoor air temperature is measured
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)  
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=self.iddata.samplingTime)
    

class Canonical(BaseBlackBox):

    def __init__(self):
        super().__init__()
        self.n_parameters = 14
        self.n_states = 3
        
        self._generate_bounds()
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("a31", "", 1.0),
            1: ("a32", "", 1.0),
            2: ("a33", "", 1.0),           
            3: ("b11", "", 1.0),
            4: ("b12", "", 1.0),
            5: ("b13", "", 1.0),
            6: ("b21", "", 1.0),
            7: ("b22", "", 1.0),
            8: ("b23", "", 1.0),
            9: ("b31", "", 1.0),
            10: ("b32", "", 1.0),
            11: ("b33", "", 1.0),
            12: ("x2[0]", "Celsius", 1.0),
            13: ("x3[0]", "Celsius", 1.0)
        }
    
    
    def create_model(self, par):  
        
        A = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [par[0], par[1], par[2]]
            ])
        
        B = np.array([
            [par[3], par[4], par[5]],
            [par[6], par[7], par[8]],
            [par[9], par[10], par[11]]
            ])
        
        C = np.array([[1, 0, 0]])
        D = np.array([[0, 0, 0]])
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[12]],
                       [par[13]]])
        
        K = self.feedback_matrix(par)  
 
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=self.iddata.samplingTime)