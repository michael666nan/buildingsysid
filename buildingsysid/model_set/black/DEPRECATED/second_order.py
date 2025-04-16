import numpy as np

from system_identification.black.base_blackbox import BaseBlackBox
from system_identification.statespace import StateSpace



# =================================================================
# Full Model - NOT IDENTIFIABLE
# =================================================================
class Full(BaseBlackBox):

    def __init__(self):
        super().__init__()
        self.n_parameters = 11
        self.n_states = 2
        
        self._generate_bounds()
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("a11", "", 1.0),
            1: ("a12", "", 1.0),
            2: ("a21", "", 1.0),
            3: ("a22", "", 1.0),
            4: ("b11", "", 1.0),
            5: ("b12", "", 1.0),
            6: ("b13", "", 1.0),
            7: ("b21", "", 1.0),
            8: ("b22", "", 1.0),
            9: ("b23", "", 1.0),
            10: ("x1[0]", "", 1.0)
        }
    
    
    def create_model(self, par):
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[10]]])
        
        # Discrete-time state-space matrices
        A = np.array([
            [par[0], par[1]],
            [par[2], par[3]]
        ])
        
        B = np.array([
            [par[4], par[5], par[6]],
            [par[7], par[8], par[9]]
            
        ])
        
        C = np.array([[1, 0]])  # Only indoor air temperature is measured
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)  
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=self.iddata.samplingTime)
    

class Canonical(BaseBlackBox):

    def __init__(self):
        super().__init__()
        self.n_parameters = 9
        self.n_states = 2
        
        self._generate_bounds()
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("a21", "", 1.0),
            1: ("a22", "", 1.0),
            2: ("b11", "", 1.0),
            3: ("b12", "", 1.0),
            4: ("b13", "", 1.0),
            5: ("b21", "", 1.0),
            6: ("b22", "", 1.0),
            7: ("b23", "", 1.0),
            8: ("x2[0]", "", 1.0)
        }
    
    
    def create_model(self, par):
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[8]]])
        
        # Continuous-time state-space matrices
        A = np.array([
            [0, 1],
            [par[0], par[1]]
        ])
        
        B = np.array([
            [par[2], par[3], par[4]],
            [par[5], par[6], par[7]]
            
        ])
        
        C = np.array([[1, 0]])
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par) 
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=self.iddata.samplingTime)