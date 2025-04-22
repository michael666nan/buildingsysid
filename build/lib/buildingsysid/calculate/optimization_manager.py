import numpy as np
import warnings
import copy

from buildingsysid.calculate.solvers.least_squares_solver import LeastSquaresSolver
from buildingsysid.criterion_of_fit.objective_functions import StandardObjective

class OptimizationManager:
    """
    Coordinator class for system identification optimization problems
    with enhanced error handling and robustness.
    
    This class coordinates the optimization process by:
    - Connecting the model with the objective function and data
    - Setting up the bounds for optimization
    - Preparing the problem for external solvers
    - Handling numerical issues during optimization
    """
    
    def __init__(self, model_structure, data, objective=None, solver=None):
        """
        Initialize the optimization problem.
        
        Args:
            model_structure: Instance of model structure
            data: IdData object with the data to use for identification
            objective: ObjectiveFunction instance (if None, it must be set later)
            solver: Solver instance (if None, a default LeastSquaresSolver will be created)
        """
        
        # Create a working copy of the model structure
        self.model = copy.deepcopy(model_structure)
        
        # Set the iddata
        self.data = data
        
        # Set the objective
        self.objective = objective if objective is not None else StandardObjective()
        
        # Set the solver
        self.solver = solver if solver is not None else LeastSquaresSolver()
        
        # Configure feedback parameters if needed
        if hasattr(self.objective, 'kstep'):
            # If objective uses feedback (kstep > 0), enable feedback parameters
            self.model.set_feedback_mode(use_feedback=self.objective.kstep > 0)
        
        # Results storage
        self.result = None
        self.attempts = 0  # Track the number of attempts made
        self.max_attempts = 5  # Maximum number of attempts to try
    
    
    # =================================================================
    # Method to Wrap Around the Objective (To Fit the Solver)
    # =================================================================    
    def objective_wrapper(self, par):
        """
        Wrapper around the objective function that handles numerical issues.
        
        Args:
            par: Parameter vector
            
        Returns:
            Array of residuals from the objective function
        """
        if self.objective is None:
            raise ValueError("No objective function specified")
        
        try:
            # Try to evaluate the objective function
            residuals = self.objective(self.model, self.data, par)
            
            # Check for NaN or Inf values
            if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
                # Return a high penalty value instead of NaN/Inf
                return np.ones_like(residuals) * 1e10
            
            return residuals
        except Exception as e:
            # If there's an error, return a high penalty value
            print(f"Warning: Error in objective function: {e}")
            # Return a large residual vector
            return np.ones(self.data.y.size) * 1e10
    
    
    # =================================================================
    # Method to Calculate RMSE from Residuals
    # =================================================================
    def calculate_rmse(self, residuals):
        """
        Calculate RMSE directly from residuals.
        
        Args:
            residuals: Vector of residuals
            
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean(residuals**2))
    
    
    # =================================================================
    # Method to Generate Initial Guesses
    # =================================================================    
    def generate_initial_parameters(self, strategy='random'):
        """
        Generate initial parameters using various strategies.
        
        Args:
            strategy: Strategy to use ('random', 'zeros', 'ones', 'middle', 'perturbed', 'biased_random')
                    or a numpy array/list of specific initial values
            
        Returns:
            Array of initial parameter values
        """
        # Check if strategy is already a parameter vector
        if isinstance(strategy, (np.ndarray, list)):
            # Convert to numpy array if needed
            return np.array(strategy)
        
        # Get bounds
        lower_bounds, upper_bounds = self.model.get_parameter_bounds()
        
        # Number of free parameters
        n_params = self.model.n_free_parameters
        
        if strategy == 'random':
            # Random initialization (uniform distribution between sensible bounds)
            lower_init = np.where(np.isfinite(lower_bounds), lower_bounds, -1.0)
            upper_init = np.where(np.isfinite(upper_bounds), upper_bounds, 1.0)
            
            # Add small margin to avoid boundary values
            margin = 0.05 * (upper_init - lower_init)
            lower_init = lower_init + margin
            upper_init = upper_init - margin
            
            result = np.random.uniform(low=lower_init, high=upper_init, size=n_params)
        
        elif strategy == 'zeros':
            result = np.zeros(n_params)
            
            # Check if zeros are within bounds and adjust if needed
            for i in range(n_params):
                if result[i] < lower_bounds[i]:
                    result[i] = lower_bounds[i]
                elif result[i] > upper_bounds[i]:
                    result[i] = upper_bounds[i]
        
        elif strategy == 'ones':
            result = np.ones(n_params)
            
            # Check if ones are within bounds and adjust if needed
            for i in range(n_params):
                if result[i] < lower_bounds[i]:
                    result[i] = lower_bounds[i]
                elif result[i] > upper_bounds[i]:
                    result[i] = upper_bounds[i]
        
        elif strategy == 'middle':
            # Middle of bounds range
            lower_init = np.where(np.isfinite(lower_bounds), lower_bounds, -1.0)
            upper_init = np.where(np.isfinite(upper_bounds), upper_bounds, 1.0)
            result = 0.5 * (lower_init + upper_init)
        
        elif strategy == 'perturbed':
            # Start with zeros but add small perturbations
            result = np.random.normal(0, 0.1, size=n_params)
            
            # Check bounds
            for i in range(n_params):
                if result[i] < lower_bounds[i]:
                    result[i] = lower_bounds[i]
                elif result[i] > upper_bounds[i]:
                    result[i] = upper_bounds[i]
        
        elif strategy == 'black_box':
            # Random values biased toward typical values but respecting bounds
            result = np.zeros(n_params)
            
            for i in range(n_params):
                idx = self.model.free_indices[i]
                param_name = self.model.param_dict[idx][0]
                
                # Get bounds for this parameter
                lower = lower_bounds[i]
                upper = upper_bounds[i]
                
                # Ensure bounds are finite for sampling
                if not np.isfinite(lower):
                    lower = -1.0
                if not np.isfinite(upper):
                    upper = 1.0
                    
                # Bias initial guesses based on parameter type
                if param_name.startswith('a'):  # State matrix elements
                    # Dynamic parameters often negative for stable systems
                    proposed = np.random.uniform(-1.0, -0.1)
                elif param_name.startswith('b'):  # Input matrix elements
                    # Input gains often small positive
                    proposed = np.random.uniform(0.1, 1.0)
                elif param_name.startswith('k'):  # Feedback matrix elements
                    # Feedback parameters are often small
                    proposed = np.random.uniform(-0.2, 0.2)
                else:
                    # Default random
                    proposed = np.random.uniform(-1.0, 1.0)
                
                # Enforce bounds
                result[i] = np.clip(proposed, lower, upper)
                
        elif strategy == 'grey_box':
            result = np.zeros(n_params)
            
            for i in range(n_params):
                idx = self.model.free_indices[i]
                param_name = self.model.param_dict[idx][0]
                
                # Get bounds for this parameter
                lower = lower_bounds[i]
                upper = upper_bounds[i]
                
                # Ensure bounds are finite for sampling
                if not np.isfinite(lower):
                    lower = -1.0
                if not np.isfinite(upper):
                    upper = 1.0
                    
                # Physics-informed initial values
                if param_name.startswith('H'):  # specific heat transfer
                    proposed = np.random.uniform(0.1, 1.0)
                elif param_name.startswith('C'):  # specific heat capacity
                    proposed = np.random.uniform(0.1, 1.0)
                elif param_name.startswith('ws'):  # solar input
                    proposed = np.random.uniform(0.1, 1.0)
                elif param_name.startswith('wh'):  # heat input
                    proposed = np.random.uniform(0.1, 1.0)
                elif param_name.startswith('k'):  # Feedback matrix elements
                    proposed = np.random.uniform(-0.2, 0.2)
                elif param_name.startswith('x'):  # Initial state (temperatures)
                    proposed = np.random.uniform(18, 22)
                else:
                    # Default random
                    proposed = np.random.uniform(-1.0, 1.0)
                
                # Enforce bounds
                result[i] = np.clip(proposed, lower, upper)
        
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
            
        return result
    
    
    # =================================================================
    # Method to Solve Optimization Problem
    # =================================================================     
    def solve(self, x0=None, max_attempts=5, initialization_strategies=["random"], max_rmse=None):
        """
        Solve the optimization problem using the provided solver with robustness.
        
        Args:
            x0: Initial parameter values (if None, generated automatically)
            max_attempts: Maximum number of attempts per strategy if the strategy fails
            initialization_strategies: List of strategies to try.
                Can include: 'random', 'zeros', 'middle', 'perturbed', 'ones', 'biased_random'
                or numpy arrays of specific initial values.
                Each strategy will be tried until success or max_attempts is reached.
                All strategies will be tried regardless of max_attempts.
            max_rmse: Maximum acceptable RMSE (if None, continues all attempts)
                Solutions with RMSE below this threshold are considered satisfactory.
            
        Returns:
            A new, updated model structure instance (the original model is not modified)
        """
        
        # Get bounds
        lower_bounds, upper_bounds = self.model.get_parameter_bounds()
        
        # Set maximum attempts per strategy
        self.max_attempts = max_attempts
        
        strategies = initialization_strategies
        
        # If x0 is provided, add it as the first strategy
        if x0 is not None:
            strategies = [x0] + list(strategies)
        
        # Try optimization with different initial parameters
        best_result = None
        lowest_cost = float('inf')
        lowest_rmse = float('inf')
        self.attempts = 0
        
        # Iterate through all strategies
        for strategy_idx, strategy in enumerate(strategies):
            strategy_success = False
            strategy_attempts = 0
            
            # Determine the strategy name for display purposes
            if isinstance(strategy, (np.ndarray, list)):
                strategy_name = "custom array"
            else:
                strategy_name = strategy
            
            print(f"\nTrying strategy {strategy_idx+1}/{len(strategies)}: {strategy_name}")
            
            # Try this strategy up to max_attempts times
            while not strategy_success and strategy_attempts < max_attempts:
                strategy_attempts += 1
                self.attempts += 1
                
                # Generate initial parameters for this attempt
                x0_attempt = self.generate_initial_parameters(strategy)
                
                print(f"  Attempt {strategy_attempts}/{max_attempts} with {strategy_name} initialization")
                
                try:
                    # Catch warnings during optimization
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        
                        # Call the solver with the problem components
                        result = self.solver.solve(
                            objective_fn=self.objective_wrapper,
                            x0=x0_attempt, 
                            bounds=(lower_bounds, upper_bounds)
                        )
                    
                    # Check if optimization succeeded and improved the objective
                    success = hasattr(result, 'success') and result.success
                    cost = result.cost
                    
                    # Calculate RMSE for this result directly from residuals
                    if hasattr(result, 'fun'):
                        current_rmse = self.calculate_rmse(result.fun)
                    else:
                        # Fallback in case residuals aren't available
                        current_rmse = float('inf')
                    
                    # Add RMSE to the result
                    result.rmse = current_rmse
                    
                    print(f"    Success: {success}, RMSE: {current_rmse:.6f}, Cost (SSE): {cost:.6e}")
                    
                    # Mark this strategy as successful
                    if success:
                        strategy_success = True
                        
                        # Update best result if this attempt is better
                        if cost < lowest_cost:
                            lowest_cost = cost
                            lowest_rmse = current_rmse
                            best_result = result
                            print(f"    → New best result (RMSE: {lowest_rmse:.6f})")
                            
                            # Break early if we have a good solution
                            if max_rmse is not None and current_rmse < max_rmse:
                                print(f"    → Found satisfactory solution (RMSE < {max_rmse:.6f})")
                                break
                    
                except Exception as e:
                    error_str = str(e)
                    print(f"    Optimization attempt failed: {error_str}")
                    
                    # If the error is about NaNs or Infs, show that we'll retry
                    if "infs or NaNs" in error_str:
                        print(f"    → Detected NaN/Inf error. Will retry with a new {strategy_name} initialization.")
                        # Continue the while loop to retry this strategy
        
        # Create an optimized model as a copy
        optimized_model = None
        
        # Use the best result
        if best_result is not None:
            self.result = best_result
            
            # Create a new model instance with the optimized parameters
            optimized_model = copy.deepcopy(self.model)
            
            if hasattr(best_result, 'x'):
                # Store the result in the model (but don't modify the original)
                optimized_model.result = best_result                
                print("\nOptimization completed successfully")
                print(f"Final RMSE: {lowest_rmse:.6f}")
                print(f"Final SSE: {lowest_cost:.6e}")
        else:
            print(f"\nWarning: All {self.attempts} optimization attempts across all strategies failed.")
            
            # Create a dummy result
            class FailedResult:
                def __init__(self):
                    self.success = False
                    self.message = "All optimization attempts failed"
                    self.x = None
                    self.fun = np.array([float('inf')])  # Placeholder for residuals
                    self.rmse = float('inf')
            
            self.result = FailedResult()
            # Return a copy of the original model in case of failure
            optimized_model = copy.deepcopy(self.model)
        
        return optimized_model