import numpy as np
from buildingsysid.model_set.black.base_blackbox import BaseBlackBox
from buildingsysid.utils.statespace import StateSpace

# =================================================================
# First Order - Observable Canonical
# ================================================================= 
class First(BaseBlackBox):
    def __init__(self, fixed_params={}, param_bounds={}):
        super().__init__()
        
        # Define total parameters and state dimensions
        self.total_parameters = 6
        self.n_states = 1
        
        # Define parameter dictionary
        self.param_dict = {
            0: ("a11", "", 1.0),
            1: ("b11", "", 1.0),
            2: ("b12", "", 1.0),
            3: ("b13", "", 1.0),
            4: ("k1", "", 1.0),
            5: ("x1[0]", "", 1.0)
        }
        
        self._initialize_fixed_parameters(fixed_params, param_bounds)
    
    
    def create_model(self, par):
        """ Create a first-order state-space model from the parameters. """
        # Expand parameters to get full model
        full_model_par = self._expand_parameters(par)
        
        # Extract model parameters
        a11 = full_model_par[0]
        b11 = full_model_par[1]
        b12 = full_model_par[2]
        b13 = full_model_par[3]
        k1 = full_model_par[4]
        x1_0 = full_model_par[5]
        
        x0 = np.array([[x1_0]])
        
        A = np.array([[a11]])
        
        B = np.array([[b11, b12, b13]])
        
        C = np.array([[1]])
        D = np.array([[0, 0, 0]])
        
        K = np.array([[k1]])
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=1)


# =================================================================
# Second Order - Observable Canonical
# ================================================================= 
class Second(BaseBlackBox):
    def __init__(self, fixed_params={}, param_bounds={}):
        super().__init__()
        
        # Define total parameters and state dimensions
        self.total_parameters = 12
        self.n_states = 2
        
        # Define parameter dictionary
        self.param_dict = {
            0: ("a21", "", 1.0),
            1: ("a22", "", 1.0),
            2: ("b11", "", 1.0),
            3: ("b12", "", 1.0),
            4: ("b13", "", 1.0),
            5: ("b21", "", 1.0),
            6: ("b22", "", 1.0),
            7: ("b23", "", 1.0),
            8: ("k1", "", 1.0),
            9: ("k2", "", 1.0),
            10: ("x1[0]", "", 1.0),
            11: ("x2[0]", "", 1.0),
        }

        self._initialize_fixed_parameters(fixed_params, param_bounds)
    
    def create_model(self, par):
        """ Create a second-order state-space model from the parameters. """
        
        # Expand parameters to get full model
        full_model_par = self._expand_parameters(par)
        
        # Extract model parameters
        a21 = full_model_par[0]
        a22 = full_model_par[1]
        b11 = full_model_par[2]
        b12 = full_model_par[3]
        b13 = full_model_par[4]
        b21 = full_model_par[5]
        b22 = full_model_par[6]
        b23 = full_model_par[7]
        k1 = full_model_par[8]
        k2 = full_model_par[9]
        x1_0 = full_model_par[10]
        x2_0 = full_model_par[11]
        
        x0 = np.array([[x1_0],
                       [x2_0]])
        
        A = np.array([[0, 1],
                      [a21, a22]])
        
        B = np.array([[b11, b12, b13],
                      [b21, b22, b23]])
        
        C = np.array([[1, 0]])
        D = np.array([[0, 0, 0]])
        
        K = np.array([[k1],
                      [k2]])
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=1)


#=================================================================
# Third Order - Observable Canonical
#================================================================= 
class Third(BaseBlackBox):
    def __init__(self, fixed_params={}, fixed_bounds={}):
        super().__init__()
        
        # Define total parameters and state dimensions
        self.total_parameters = 18
        self.n_states = 3
        
        # Define parameter dictionary
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
            12: ("k1", "", 1.0),
            13: ("k2", "", 1.0),
            14: ("k3", "", 1.0),
            15: ("x1[0]", "", 1.0),
            16: ("x2[0]", "", 1.0),
            17: ("x3[0]", "", 1.0)
        }

        self._initialize_fixed_parameters(fixed_params, param_bounds)
    
    
    def create_model(self, par):
        """ Create a third-order state-space model from the parameters. """
        
        # Expand parameters to get full model
        full_model_par = self._expand_parameters(par)
        
        # Extract model parameters
        a31 = full_model_par[0]
        a32 = full_model_par[1]
        a33 = full_model_par[2]
        b11 = full_model_par[3]
        b12 = full_model_par[4]
        b13 = full_model_par[5]
        b21 = full_model_par[6]
        b22 = full_model_par[7]
        b23 = full_model_par[8]
        b31 = full_model_par[9]
        b32 = full_model_par[10]
        b33 = full_model_par[11]
        k1 = full_model_par[12]
        k2 = full_model_par[13]
        k3 = full_model_par[14]
        x1_0 = full_model_par[15]
        x2_0 = full_model_par[16]
        x3_0 = full_model_par[17]
        
        x0 = np.array([[x1_0],
                       [x2_0],
                       [x3_0]])
        
        A = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [a31, a32, a33]
        ])
        
        B = np.array([
            [b11, b12, b13],
            [b21, b22, b23],
            [b31, b32, b33]
        ])
        
        C = np.array([[1, 0, 0]])
        D = np.array([[0, 0, 0]])
        
        K = np.array([[k1],
                      [k2],
                      [k3]])
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=1)


# =================================================================
# Fourth Order - Observable Canonical
# ================================================================= 
class Fourth(BaseBlackBox):
    def __init__(self, fixed_params={}, param_bounds={}):
        super().__init__()
        
        # Define total parameters and state dimensions
        self.total_parameters = 24
        self.n_states = 4
        
        # Define parameter dictionary
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
            16: ("k1", "", 1.0),
            17: ("k2", "", 1.0),
            18: ("k3", "", 1.0),
            19: ("k4", "", 1.0),
            20: ("x1[0]", "", 1.0),
            21: ("x2[0]", "", 1.0),
            22: ("x3[0]", "", 1.0),
            23: ("x4[0]", "", 1.0)
        }
        
        self._initialize_fixed_parameters(fixed_params, param_bounds)
        
    
    def create_model(self, par):
    
        """ Create a fourth-order state-space model from the parameters. """
        
        # Expand parameters to get full model
        full_model_par = self._expand_parameters(par)
        
        # Extract model parameters
        a41 = full_model_par[0]
        a42 = full_model_par[1]
        a43 = full_model_par[2]
        a44 = full_model_par[3]
        b11 = full_model_par[4]
        b12 = full_model_par[5]
        b13 = full_model_par[6]
        b21 = full_model_par[7]
        b22 = full_model_par[8]
        b23 = full_model_par[9]
        b31 = full_model_par[10]
        b32 = full_model_par[11]
        b33 = full_model_par[12]
        b41 = full_model_par[13]
        b42 = full_model_par[14]
        b43 = full_model_par[15]
        k1 = full_model_par[16]
        k2 = full_model_par[17]
        k3 = full_model_par[18]
        k4 = full_model_par[19]
        x1_0 = full_model_par[20]
        x2_0 = full_model_par[21]
        x3_0 = full_model_par[22]
        x4_0 = full_model_par[23]
        
        x0 = np.array([[x1_0],
                       [x2_0],
                       [x3_0],
                       [x4_0]])
        
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [a41, a42, a43, a44]
        ])
        
        B = np.array([
            [b11, b12, b13],
            [b21, b22, b23],
            [b31, b32, b33],
            [b41, b42, b43]
        ])
        
        C = np.array([[1, 0, 0, 0]])
        D = np.array([[0, 0, 0]])
        
        K = np.array([[k1],
                      [k2],
                      [k3],
                      [k4]])
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=1)