import numpy as np
from scipy.optimize import least_squares, differential_evolution, minimize


# =====================================================================
# Define solvers - these would typically be in a separate module
# =====================================================================

def least_squares_solver(objective_fn, x0, bounds, options=None):
    """
    Solver using scipy's least_squares optimizer.
    
    Args:
        objective_fn: Function that calculates residuals
        x0: Initial parameter values
        bounds: Tuple of (lower_bounds, upper_bounds)
        options: Dictionary of solver options
        
    Returns:
        Optimization result object
    """
    options = options or {}
    method = options.pop('method', 'trf')
    verbose = options.pop('verbose', 1)
    
    # Run optimization
    result = least_squares(
        objective_fn,
        x0,
        bounds=bounds,
        method=method,
        verbose=verbose,
        **options
    )
    
    return result


def global_optimizer_solver(objective_fn, x0, bounds, options=None):
    """
    Solver using differential evolution for global optimization.
    
    Args:
        objective_fn: Function that calculates residuals
        x0: Initial parameter values (not used by differential evolution)
        bounds: Tuple of (lower_bounds, upper_bounds)
        options: Dictionary of solver options
        
    Returns:
        Optimization result object
    """
    options = options or {}
    popsize = options.pop('popsize', 15)
    maxiter = options.pop('maxiter', 100)
    disp = options.pop('disp', True)
    
    # Prepare bounds for differential_evolution
    de_bounds = list(zip(bounds[0], bounds[1]))
    
    # Define sum of squares objective
    def sum_sq_objective(x):
        residuals = objective_fn(x)
        return np.sum(residuals**2)
    
    # Run optimization
    result = differential_evolution(
        sum_sq_objective,
        de_bounds,
        popsize=popsize,
        maxiter=maxiter,
        disp=disp,
        **options
    )
    
    return result


def custom_solver(objective_fn, x0, bounds, options=None):
    """
    Custom solver implementation using a simple gradient-free approach.
    This is just a simple example - a real implementation would be more sophisticated.
    
    Args:
        objective_fn: Function that calculates residuals
        x0: Initial parameter values
        bounds: Tuple of (lower_bounds, upper_bounds)
        options: Dictionary of solver options
        
    Returns:
        Object with optimization results
    """
    options = options or {}
    max_iter = options.get('max_iter', 100)
    step_size = options.get('step_size', 0.1)
    tolerance = options.get('tolerance', 1e-6)
    
    # Initialize best solution
    x_best = np.array(x0, dtype=float)
    f_best = np.sum(objective_fn(x_best)**2)  # Sum of squares
    
    # Track progress
    iterations = 0
    converged = False
    
    # Simple coordinate descent algorithm
    while iterations < max_iter and not converged:
        iterations += 1
        x_prev = x_best.copy()
        
        # Try to improve each parameter
        for i in range(len(x_best)):
            # Try increasing
            x_try = x_best.copy()
            x_try[i] += step_size
            # Ensure within bounds
            x_try[i] = min(x_try[i], bounds[1][i])
            
            # Evaluate
            f_try = np.sum(objective_fn(x_try)**2)
            if f_try < f_best:
                x_best = x_try
                f_best = f_try
                continue
                
            # Try decreasing
            x_try = x_best.copy()
            x_try[i] -= step_size
            # Ensure within bounds
            x_try[i] = max(x_try[i], bounds[0][i])
            
            # Evaluate
            f_try = np.sum(objective_fn(x_try)**2)
            if f_try < f_best:
                x_best = x_try
                f_best = f_try
        
        # Check convergence
        change = np.linalg.norm(x_best - x_prev)
        if change < tolerance:
            converged = True
            
        # Print progress
        if iterations % 10 == 0 or converged:
            print(f"Iteration {iterations}: Best SSE = {f_best:.6f}, Change = {change:.6f}")
    
    # Create a result object similar to scipy's
    class OptimResult:
        pass
    
    result = OptimResult()
    result.x = x_best
    result.fun = objective_fn(x_best)
    result.success = converged
    result.message = "Converged" if converged else "Maximum iterations reached"
    result.nit = iterations
    
    return result


