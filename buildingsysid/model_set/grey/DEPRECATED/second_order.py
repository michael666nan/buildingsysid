import numpy as np

from criterion_of_fit.base_greybox import BaseGreyBox
from criterion_of_fit.statespace import StateSpace


# =================================================================
# Full Model - NOT IDENTIFIABLE
# =================================================================
class Full(BaseGreyBox):
    
    def __init__(self, floor_area):
        super().__init__(floor_area)
        self.n_parameters = 10
        self.n_states = 2
        
        self._generate_bounds()    
    
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("H_im", "W/(K*m²)", self.floor_area),
            1: ("H_ia", "W/(K*m²)", self.floor_area),
            2: ("H_ma", "W/(K*m²)", self.floor_area),
            3: ("C_i", "Wh/(K*m²)", self.floor_area * 3600),
            4: ("C_m", "Wh/(K*m²)", self.floor_area * 3600),
            5: ("ws1", "(m²)", 1.0),
            6: ("ws2", "m²", 1.0),
            7: ("wh1", "-", 1.0),
            8: ("wh2", "-", 1.0),
            9: ("Tm[0]", "", 1.0)
        }
    
    def create_model(self, par):
        # Use the scales defined in param_dict
        H_im = par[0] * self.param_dict[0][2]  # (W/K)
        H_ia = par[1] * self.param_dict[1][2]  # (W/K)
        H_ma = par[2] * self.param_dict[2][2]  # (W/K)
        C_i = par[3] * self.param_dict[3][2]   # (J/K)
        C_m = par[4] * self.param_dict[4][2]   # (J/K)
        ws1 = par[5] * self.param_dict[5][2]   # (m²)
        ws2 = par[6] * self.param_dict[6][2]   # (m²) 
        wh1 = par[7] * self.param_dict[7][2]   # (-)
        wh2 = par[8] * self.param_dict[8][2]   # (-)

        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[9]]])

        # Calculate the elements of the matrix
        a11 = -(H_ia + H_im) / C_i
        a12 = H_im / C_i
        a21 = H_im / C_m
        a22 = -(H_ma + H_im) / C_m
        
        # Construct the matrix A
        A = np.array([
            [a11, a12],
            [a21, a22]
        ])
                
        B = np.array([
            [H_ia/C_i, ws1/C_i, wh1/C_i],
            [H_ma/C_m, ws2/C_m, wh2/C_m]
            
        ])
        
        C = np.array([[1, 0]])
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)






##### IDENTIFIABLE SUBSTRUCTURES:
# =================================================================
# Assumptions:  - 100 % convection
#               - Internal Mass not connected to outdoor
#               - Solar radiation 100 % into internal mass
# =================================================================
class ConvectorInternalMass(BaseGreyBox):
    
    def __init__(self, floor_area):
        super().__init__(floor_area)
        self.n_parameters = 6
        self.n_states = 2
        
        self._generate_bounds()
        
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("H_ia", "W/(K*m²)", self.floor_area),
            1: ("H_im", "W/(K*m²)", self.floor_area),
            2: ("C_i", "Wh/((KK*m²)", self.floor_area * 3600),
            3: ("C_m", "Wh/(K*m²)", self.floor_area * 3600),
            4: ("gA", "-", 1.0),
            5: ("Tm[0]", "-", 1.0)
        }
        
    
    def create_model(self, par):
        # Use the scales defined in param_dict
        H_ia = par[0] * self.param_dict[0][2]  # (W/K)
        H_im = par[1] * self.param_dict[1][2]  # (W/K)
        C_i = par[2] * self.param_dict[2][2]   # (J/K)
        C_m = par[3] * self.param_dict[3][2]   # (J/K)
        gA = par[4] * self.param_dict[4][2]    # (m²)
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[5]]])

        # Continuous-time state-space matrices
        A = np.array([
            [-(H_ia + H_im)/C_i, H_im/C_i],
            [H_im/C_m, -H_im/C_m]
        ])
        
        B = np.array([
            [H_ia/C_i, 0, 1/C_i],
            [0, gA/C_m, 0]
        ])
        
        C = np.array([[1, 0]])  # Only indoor air temperature is measured
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)        
        
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)
                    

        
# =================================================================
# Assumptions:  - 100 % Convection
#               - Solar radiation ws1 % into air node and ws2 into internal mass
#               - No Ground
# =================================================================
class Convector(BaseGreyBox):
    
    def __init__(self, floor_area):
        super().__init__(floor_area)
        self.n_parameters = 8
        self.n_states = 2
        
        self._generate_bounds()        
    
        # Define parameter dictionary for printing: (name, unit, scale_factor)
        self.param_dict = {
            0: ("H_im", "W/(K*m²)", self.floor_area),
            1: ("H_ia", "W/(K*m²)", self.floor_area),
            2: ("H_ma", "W/(K*m²)", self.floor_area),
            3: ("C_i", "Wh/(K*m²)", self.floor_area * 3600),
            4: ("C_m", "Wh/(K*m²)", self.floor_area * 3600),
            5: ("ws1", "(m²)", 1.0),
            6: ("ws2", "m²", 1.0),
            7: ("Tm[0]", "", 1.0)
        }
    
    def create_model(self, par):
        # Use the scales defined in param_dict
        H_im = par[0] * self.param_dict[0][2]  # (W/K)
        H_ia = par[1] * self.param_dict[1][2]  # (W/K)
        H_ma = par[2] * self.param_dict[2][2]  # (W/K)
        C_i = par[3] * self.param_dict[3][2]   # (J/K)
        C_m = par[4] * self.param_dict[4][2]   # (J/K)
        ws1 = par[5] * self.param_dict[5][2]   # (m²)
        ws2 = par[6] * self.param_dict[6][2]   # (m²) 
        
        # Initial state
        x0 = np.array([[self.iddata.y[0,0]],
                       [par[7]]])
        
        # Calculate the elements of the matrix
        a11 = -(H_ia + H_im) / C_i
        a12 = H_im / C_i
        a21 = H_im / C_m
        a22 = -(H_ma + H_im) / C_m
        
        # Construct the matrix A
        A = np.array([
            [a11, a12],
            [a21, a22]
        ])
                
        B = np.array([
            [H_ia/C_i, ws1/C_i, 1/C_i],
            [H_ma/C_m, ws2/C_m, 0]
            
        ])
        
        C = np.array([[1, 0]])
        D = np.array([[0, 0, 0]])
        
        K = self.feedback_matrix(par)
            
        return StateSpace(A, B, C, D, K, x0, samplingTime=0)