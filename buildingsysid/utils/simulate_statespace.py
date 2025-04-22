import numpy as np

from .statespace import StateSpace


def simulation(ss, iddata, kstep=-999, sum_horizon=False):

    if ss.samplingTime==0:
        ss = c2d(ss, iddata.samplingTime)
      
    # kstep negativ --> Simulation
    if kstep<=0:
        ss.K[:] = 0 

    # one step ahead predictions (kstep=1)
    y_sim, x_sim = one_step_prediction(ss, iddata)
    
    # k step ahead predictions (kstep>1)
    if kstep>1:
        y_sim = kstep_predictions(ss, iddata, x_sim, kstep, sum_horizon)
    
    return y_sim


def one_step_prediction(ss, iddata):
    
    u = iddata.u 
    y = iddata.y 
    A = ss.A 
    B = ss.B 
    C = ss.C 
    D = ss.D 
    K = ss.K 
    x0 = ss.x0
    
    #Initialize
    N = y.shape[1]
    y_sim = np.zeros((y.shape[0], N))
    x = np.zeros((A.shape[0], N))
    
    # Initial state
    x[:,0] = x0.flatten()
    
    for k in range(N-1):
        y_sim[:,k] = C @ x[:,k] + D @ u[:,k]
        error = y[:,k] - y_sim[:,k]
        x[:,k+1] = A @ x[:,k] + B @ u[:,k] + K @ error  
        
    y_sim[:,k+1] = C @ x[:,k+1] + D @ u[:,k+1]    
    
    return y_sim, x


def kstep_predictions(ss, iddata, x_one_step, kstep, sum_horizon=False):

    u = iddata.u
    y= iddata.y     
    A = ss.A 
    B = ss.B 
    C = ss.C 
    D = ss.D    
    
    ny = y.shape[0]
    N = y.shape[1]
    Y = np.zeros((kstep, N-kstep+1))
    
    # Loop through all time steps (minus the last horizon)
    for i in range(N-kstep+1):   
        
        # Assign current state (one step prediction)
        z = x_one_step[:,i]
        
        # Loop through the horizon
        for k in range(kstep-1): 
            y = C @ z + D @ u[:,i+k]
            Y[k*ny:k*ny+ny,i] = y.flatten() 
            
            z = A @ z + B @ u[:,i+k]
        
        k = k+1
        y = C @ z + D @ u[:,i+k]
        Y[k*ny:k*ny+ny,i] = y.flatten()    
    
        # Return all horizons as one big vector
        if sum_horizon:
            y_sim = Y.T.flatten().reshape((1,-1))
        
        # Or return only values at the end of the horizon
        else:
            first_column = Y[:, 0].reshape(1, -1) # Extract the first column           
            last_row_except_first = Y[-1, 1:].reshape(1, -1)  # Extract the last row, excluding the first value, and reshape it to be a 2D row vector        
            y_sim = np.hstack((first_column, last_row_except_first))             

    return y_sim


from scipy.signal import cont2discrete

def c2d(ss, samplingTime):
    A, B, C, D, _ = cont2discrete((ss.A, ss.B, ss.C, ss.D), samplingTime, method='zoh')
    
    return StateSpace(A, B, C, D, ss.K, ss.x0, samplingTime)



