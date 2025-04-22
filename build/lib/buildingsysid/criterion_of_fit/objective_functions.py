import numpy as np

from buildingsysid.utils.simulate_statespace import simulation
from buildingsysid.utils.hankel import hankel

class ObjectiveFunction:
    """
    Base class for objective functions used in system identification.
    
    This class encapsulates the logic for calculating the residuals
    between measured and simulated/predicted outputs.
    """
    
    def __init__(self, kstep=-1, sum_hor=False, **kwargs):
        """
        Initialize the objective function.
        
        Args:
            kstep: Number of steps ahead for prediction
                   negative or zero means simulation (no feedback)
            sum_hor: Whether to use summarized horizon
            **kwargs: Additional parameters specific to the objective function
        """
        self.kstep = kstep
        self.sum_hor = sum_hor
        self.params = kwargs
        
    def __call__(self, model, data, par):
        """
        Calculate residuals for the given model, data, and parameters.
        
        This method should be overridden by subclasses.
        
        Args:
            model: The model for which to calculate residuals
            data: The data for evaluating the model
            par: Parameter vector
            
        Returns:
            Array of residuals
        """
        raise NotImplementedError("Subclasses must implement this method")


class StandardObjective(ObjectiveFunction):
    """
    Standard RMSE objective function for system identification.
    """
    
    def __call__(self, model, data, par):
        """
        Calculate standard residuals between measured and simulated/predicted outputs.
        
        Args:
            model: The model for which to calculate residuals
            data: The data for evaluating the model
            par: Parameter vector
            
        Returns:
            Array of residuals (y_measured - y_simulated)
        """
        # Create model with the parameters
        ss = model.create_model(par)
        
        # Run simulation or prediction with kstep and sum_horizon
        y_sim = simulation(ss, data, kstep=self.kstep, sum_horizon=self.sum_hor)
        
        # In case of summarized horizon --> measured output must be transformed
        if self.sum_hor and self.kstep > 0:
            Y = hankel(self.kstep, data.y)
            y_mea = Y.T.flatten().reshape((1,-1))
        else:
            y_mea = data.y
        
        # Return residuals
        return y_mea[0,:] - y_sim[0,:]


class WeightedObjective(StandardObjective):
    """
    Weighted objective function that applies weights to different parts of the residual.
    """
    
    def __init__(self, kstep=-1, sum_hor=False, weights=None, **kwargs):
        """
        Initialize the weighted objective function.
        
        Args:
            kstep: Number of steps ahead for prediction
            sum_hor: Whether to use summarized horizon
            weights: Array of weights to apply to residuals
                     If None, uniform weights are used
            **kwargs: Additional parameters
        """
        super().__init__(kstep, sum_hor, **kwargs)
        self.weights = weights
    
    def __call__(self, model, data, par):
        """
        Calculate weighted residuals.
        
        Args:
            model: The model for which to calculate residuals
            data: The data for evaluating the model
            par: Parameter vector
            
        Returns:
            Array of weighted residuals
        """
        # Get standard residuals first
        residuals = super().__call__(model, data, par)
        
        # Apply weights if provided
        if self.weights is None:
            # Default to uniform weights
            return residuals
        else:
            # Ensure weights are the right length or can be broadcast
            if len(self.weights) != len(residuals):
                raise ValueError(f"Weights length ({len(self.weights)}) doesn't match residuals length ({len(residuals)})")
            
            # Apply weights
            return residuals * self.weights


class RegularizedObjective(StandardObjective):
    """
    Objective function with Tikhonov regularization to penalize parameter deviation.
    """
    
    def __init__(self, kstep=-1, sum_hor=False, lambda_reg=0.01, target_values=None, **kwargs):
        """
        Initialize the regularized objective function.
        
        Args:
            kstep: Number of steps ahead for prediction
            sum_hor: Whether to use summarized horizon
            lambda_reg: Regularization weight
            target_values: Target parameter values for regularization
                           If None, regularization pulls toward zero
            **kwargs: Additional parameters
        """
        super().__init__(kstep, sum_hor, **kwargs)
        self.lambda_reg = lambda_reg
        self.target_values = target_values
    
    def __call__(self, model, data, par):
        """
        Calculate residuals with regularization terms.
        
        Args:
            model: The model for which to calculate residuals
            data: The data for evaluating the model
            par: Parameter vector
            
        Returns:
            Array of residuals including regularization terms
        """
        # Get standard residuals
        std_residuals = super().__call__(model, data, par)
        
        # Set up target values if not provided
        if self.target_values is None:
            target_values = np.zeros_like(par)
        else:
            target_values = self.target_values
        
        # Calculate regularization terms (parameter deviation from targets)
        reg_terms = self.lambda_reg * (par - target_values)
        
        # Combine standard residuals and regularization terms
        return np.concatenate([std_residuals, reg_terms])


class FrequencyDomainObjective(ObjectiveFunction):
    """
    Objective function that evaluates the model fit in the frequency domain.
    """
    
    def __init__(self, kstep=-1, sum_hor=False, freq_range=None, **kwargs):
        """
        Initialize the frequency domain objective function.
        
        Args:
            kstep: Number of steps ahead for prediction
            sum_hor: Whether to use summarized horizon
            freq_range: Tuple of (min_freq, max_freq) to focus on,
                        or None to use full spectrum
            **kwargs: Additional parameters
        """
        super().__init__(kstep, sum_hor, **kwargs)
        self.freq_range = freq_range
    
    def __call__(self, model, data, par):
        """
        Calculate frequency domain residuals.
        
        Args:
            model: The model for which to calculate residuals
            data: The data for evaluating the model
            par: Parameter vector
            
        Returns:
            Array of frequency domain residuals
        """
        # First get time domain residuals
        std_obj = StandardObjective(self.kstep, self.sum_hor)
        time_residuals = std_obj(model, data, par)
        
        # Calculate sampling frequency
        fs = 1.0 / data.samplingTime if hasattr(data, 'samplingTime') else 1.0
        
        # Compute FFT of residuals
        fft_residuals = np.fft.rfft(time_residuals)
        freqs = np.fft.rfftfreq(len(time_residuals), d=1.0/fs)
        
        # If frequency range specified, focus only on that part
        if self.freq_range is not None:
            min_freq, max_freq = self.freq_range
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            fft_residuals = fft_residuals[mask]
        
        # Return magnitude of frequency domain residuals
        # Converting back to real-valued array for optimizer compatibility
        return np.abs(fft_residuals)


class MultiOutputObjective(ObjectiveFunction):
    """
    Objective function that handles multiple outputs with different weights.
    """
    
    def __init__(self, kstep=-1, sum_hor=False, output_weights=None, **kwargs):
        """
        Initialize the multi-output objective function.
        
        Args:
            kstep: Number of steps ahead for prediction
            sum_hor: Whether to use summarized horizon
            output_weights: Dictionary mapping output indices to weights
                            If None, all outputs are weighted equally
            **kwargs: Additional parameters
        """
        super().__init__(kstep, sum_hor, **kwargs)
        self.output_weights = output_weights
    
    def __call__(self, model, data, par):
        """
        Calculate residuals for multiple outputs.
        
        Args:
            model: The model for which to calculate residuals
            data: The data for evaluating the model
            par: Parameter vector
            
        Returns:
            Array of combined residuals from all outputs
        """
        # Create Model with the parameters
        ss = model.create_model(par)
        
        # Run simulation
        y_sim = simulation(ss, data, kstep=self.kstep, sum_horizon=self.sum_hor)
        
        # Get measured output
        if self.sum_hor and self.kstep > 0:
            Y = data.hankel(self.kstep)
            y_mea = Y.T.flatten().reshape((1,-1))
        else:
            y_mea = data.y
        
        # Calculate residuals for each output
        all_residuals = []
        
        for i in range(y_mea.shape[0]):
            output_residuals = y_mea[i,:] - y_sim[i,:]
            
            # Apply output weight if specified
            if self.output_weights and i in self.output_weights:
                output_residuals = output_residuals * self.output_weights[i]
                
            all_residuals.append(output_residuals)
        
        # Flatten all residuals into a single array
        return np.concatenate(all_residuals)