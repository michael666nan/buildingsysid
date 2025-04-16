import numpy as np
from scipy.optimize import least_squares
import copy
    
def pem(
        model_structure, 
        iddata, 
        kstep=-1, 
        initial_guess=None, 
        sum_horizon=False, 
        ftol=1e-8, 
        max_nfev = 500,
        print_results=True, 
        plot_results=False, 
        verbose=2, 
        method='trf'):
    

    # Create a deep copy of the model to avoid modifying the original
    model_copy = copy.deepcopy(model_structure)
    
    # Assign training data
    model_copy.iddata = iddata
    
    # Cost Function Settings
    model_copy.kstep = kstep
    model_copy.sum_hor = sum_horizon
    
    # Get Number of Free Parameters
    n_params = model_copy.get_n_parameters()        
    
    # Create Initial Guess
    if initial_guess is None:
        par_ini = np.random.rand(n_params)
    else:
        par_ini=initial_guess
    
    # Create Bounds on Parameters
    my_bounds = model_copy.get_bounds()
    
    # Optimize
    if method == 'lm':
        results = least_squares(model_copy.objective, x0=par_ini, method='lm', verbose=verbose)
    else:
        results = least_squares(model_copy.objective, x0=par_ini, jac='3-point', ftol=ftol, max_nfev=max_nfev,
                               bounds=my_bounds, verbose=verbose)
        
    
    # Report
    model_copy.least_squares_report = results
    
    # Parameters
    model_copy.par = results.x
    
    # State Space Model
    model_copy.ss = model_copy.create_model(model_copy.par)
    
    # Residuals
    model_copy.residuals = results.fun
    
    # Jacobian
    model_copy.jacobian = results.jac
    
    # Confidence Intervals
    model_copy.conf_intervals = confidence_intervals(model_copy)
    
    # Outputs
    if print_results:
        model_copy.print()
    
    if plot_results:
        model_copy.plot_parameters()
    
    return model_copy


def confidence_intervals(model):
    # Parameters
    par = model.par
    
    # Residuals
    residuals = model.residuals
    
    # Jacobian
    J = model.jacobian
    
    # Estimate the Hessian Matrix
    H = J.T @ J
    
    # Estimate the Residual Variance
    NUMBER_OF_PARAMETERS = len(par)
    NUMBER_OF_OBSERVATIONS = len(residuals)
    DEGREES_OF_FREEDOM = NUMBER_OF_OBSERVATIONS - NUMBER_OF_PARAMETERS
    
    SSE = np.sum(residuals**2)
    Residual_Variance = SSE / DEGREES_OF_FREEDOM
    
    # Compute Parameter Covariance Matrix with robust inverse
    try:
        # Try standard inverse first
        Covariance_Matrix = Residual_Variance * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print("Warning: Singular Hessian matrix. Using pseudo-inverse for confidence intervals.")
        # Use pseudo-inverse as a fallback
        Covariance_Matrix = Residual_Variance * np.linalg.pinv(H)
    
    # Derive Standard Errors of Parameter Estimates
    Standard_Error = np.sqrt(np.diag(Covariance_Matrix))
    
    # Check for NaN values in Standard Error
    if np.any(np.isnan(Standard_Error)):
        print("Warning: NaN values in standard errors. Some parameters may be poorly identified.")
        # Replace NaN with infinity to indicate high uncertainty
        Standard_Error = np.where(np.isnan(Standard_Error), np.inf, Standard_Error)
    
    # Construct Confidence Interval
    z = 1.96  # 95% confidence interval
    Confidence_Interval = np.array([par - z*Standard_Error, par + z*Standard_Error])
    return Confidence_Interval