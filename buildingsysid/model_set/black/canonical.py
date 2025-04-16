import numpy as np
from buildingsysid.criterion_of_fit.base_blackbox import BaseBlackBox
from buildingsysid.criterion_of_fit.statespace import StateSpace

# =================================================================
# First Order - Observable Canonical
# ================================================================= 
class First(BaseBlackBox):
    def __init__(self):
        super().__init__()
        self.n_parameters = 4
        self.n_states = 1
        
        self._generate_bounds()
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("a11", "", 1.0),
            1: ("b11", "", 1.0),
            2: ("b12", "", 1.0),
            3: ("b13", "", 1.0)
        }
    
    def create_model(self, par):
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]]])
        
        # Continuous-time state-space matrices
        A = np.array([[par[0]]])
        
        B = np.array([[par[1], par[2], par[3]]])
        
        C = np.array([[1]])
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=self.iddata.samplingTime)


# =================================================================
# Second Order - Observable Canonical
# ================================================================= 
class Second(BaseBlackBox):
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


# =================================================================
# Third Order - Observable Canonical
# ================================================================= 
class Third(BaseBlackBox):
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
            12: ("x2[0]", "", 1.0),
            13: ("x3[0]", "", 1.0)
        }
        
        print("Model created")
    
    def create_model(self, par):
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[12]],
                       [par[13]]])
        
        # x0 = np.array([[par[12]],
        #                [self.iddata.y[0,0]],                       
        #                [par[13]]])
        
        # Continuous-time state-space matrices in observable canonical form
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
        #C = np.array([[0, 1, 0]])
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=self.iddata.samplingTime)


# =================================================================
# Fourth Order - Observable Canonical
# ================================================================= 
class Fourth(BaseBlackBox):
    def __init__(self):
        super().__init__()
        self.n_parameters = 19
        self.n_states = 4
        
        self._generate_bounds()
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("a41", "", 1.0),
            1: ("a42", "", 1.0),
            2: ("a43", "", 1.0),
            3: ("a44", "", 1.0),
            4: ("b11", "", 1.0),
            5: ("b12", "", 1.0),
            6: ("b13", "", 1.0),
            7: ("b21", "", 1.0),
            8: ("b22", "", 1.0),
            9: ("b23", "", 1.0),
            10: ("b31", "", 1.0),
            11: ("b32", "", 1.0),
            12: ("b33", "", 1.0),
            13: ("b41", "", 1.0),
            14: ("b42", "", 1.0),
            15: ("b43", "", 1.0),
            16: ("x2[0]", "", 1.0),
            17: ("x3[0]", "", 1.0),
            18: ("x4[0]", "", 1.0)
        }
    
    def create_model(self, par):
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[16]],
                       [par[17]],
                       [par[18]]])
        
        # Continuous-time state-space matrices in observable canonical form
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [par[0], par[1], par[2], par[3]]
        ])
        
        B = np.array([
            [par[4], par[5], par[6]],
            [par[7], par[8], par[9]],
            [par[10], par[11], par[12]],
            [par[13], par[14], par[15]]
        ])
        
        C = np.array([[1, 0, 0, 0]])
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=self.iddata.samplingTime)