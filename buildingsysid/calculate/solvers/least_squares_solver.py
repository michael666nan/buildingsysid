import numpy as np
from scipy.optimize import least_squares


class LeastSquaresSolver:
    """
    A class-based implementation of the least squares solver with
    support for confidence interval calculation and robust optimization.
    """
    
    def __init__(self, method='trf', verbose=0, calc_confidence=True, **options):
        """
        Initialize the solver with configuration options.
        
        Args:
            method: Optimization method - 'trf', 'dogbox', or 'lm' (default: 'trf')
            verbose: Level of verbosity (0=silent, 1=normal, 2=detailed) (default: 0)
            calc_confidence: Whether to calculate confidence intervals (default: True)
            **options: Additional options to pass to scipy.optimize.least_squares, such as:
                - ftol: Tolerance for termination by change of cost function (default: 1e-8)
                - xtol: Tolerance for termination by change of the independent variables (default: 1e-8)
                - gtol: Tolerance for termination by the norm of the gradient (default: 1e-8)
                - x_scale: Characteristic scale of each variable (default: 1.0)
                - loss: Loss function to use (default: 'linear')
                - f_scale: Value of soft margin between inlier and outlier residuals (default: 1.0)
                - max_nfev: Maximum number of function evaluations (default: None)
                - diff_step: Step size for numerical differentiation (default: None)
                - tr_solver: Method to solve trust-region subproblems (default: None)
                - tr_options: Options for trust-region solver (default: None)
                - jac_sparsity: Sparsity structure of the Jacobian (default: None)
        """
        self.method = method
        self.verbose = verbose
        self.calc_confidence = calc_confidence
        self.options = options
        self.result = None
    
    def solve(self, objective_fn, x0, bounds=None):
        """
        Solve the optimization problem using scipy's least_squares.
        
        Args:
            objective_fn: Function that calculates residuals
            x0: Initial parameter values
            bounds: Tuple of (lower_bounds, upper_bounds)
            
        Returns:
            Result from the least_squares optimization
        """
        # Check if bounds are compatible with the method
        if bounds is not None and self.method == 'lm' and (not np.all(np.isneginf(bounds[0])) or not np.all(np.isposinf(bounds[1]))):
            print("Warning: 'lm' method doesn't support bounds. Switching to 'trf' method.")
            self.method = 'trf'
        
        # For 'lm' method, bounds are ignored
        if self.method == 'lm':
            self.result = least_squares(
                objective_fn,
                x0,
                method=self.method,
                verbose=self.verbose,
                **self.options
            )
        else:
            # For 'trf' and 'dogbox' methods, include bounds
            self.result = least_squares(
                objective_fn,
                x0,
                bounds=bounds,
                method=self.method,
                verbose=self.verbose,
                **self.options
            )
        
        # Calculate confidence intervals if requested
        if self.calc_confidence:
            try:
                self._calculate_confidence_intervals()
            except Exception as e:
                print(f"Could not calculate confidence intervals: {e}")
                self.result.confidence_intervals = None
        
        return self.result
    
    def _calculate_confidence_intervals(self):
        """
        Calculate confidence intervals for the optimized parameters.
        Internal method called automatically after optimization if calc_confidence is True.
        """
        # Parameters
        par = self.result.x
        
        # Residuals
        residuals = self.result.fun
        
        # Jacobian
        J = self.result.jac
        
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
        self.result.confidence_intervals = np.array([par - z*Standard_Error, par + z*Standard_Error])
    
    def get_result(self):
        """
        Get the optimization result.
        
        Returns:
            The optimization result or None if solve() hasn't been called
        """
        return self.result
    
    def get_confidence_intervals(self):
        """
        Get the calculated confidence intervals.
        
        Returns:
            The confidence intervals array, or None if not calculated
        """
        if self.result is None:
            return None
        
        return getattr(self.result, 'confidence_intervals', None)
    
    def update_options(self, **options):
        """
        Update solver options.
        
        Args:
            **options: New options to update
            
        Returns:
            Self for method chaining
        """
        self.options.update(options)
        return self
    
    def set_method(self, method):
        """
        Set the optimization method.
        
        Args:
            method: Method to use ('trf', 'dogbox', or 'lm')
            
        Returns:
            Self for method chaining
        """
        self.method = method
        return self
    
    def set_verbose(self, verbose):
        """
        Set the verbosity level.
        
        Args:
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Self for method chaining
        """
        self.verbose = verbose
        return self
    
    def set_calc_confidence(self, calc_confidence):
        """
        Set whether to calculate confidence intervals.
        
        Args:
            calc_confidence: Whether to calculate confidence intervals
            
        Returns:
            Self for method chaining
        """
        self.calc_confidence = calc_confidence
        return self
