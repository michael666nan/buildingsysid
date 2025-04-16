import numpy as np

class IDData:
    """Base IDData class with core functionality"""
    # Core implementation

    def __init__(self, y, u, samplingTime, timestamps, 
                 y_names=None, u_names=None, y_units=None, u_units=None):
        """
        Initialize IDData object for system identification data.
        
        Args:
            y (numpy.ndarray): Output data with shape (num_outputs, num_samples) or (num_samples,)
            u (numpy.ndarray): Input data with shape (num_inputs, num_samples) or (num_samples,)
            samplingTime (float): Sampling time in seconds
            timestamps (array-like): Timestamps for the data points
            y_names (list, optional): Names for each output channel
            u_names (list, optional): Names for each input channel
            y_units (list, optional): Units for each output channel
            u_units (list, optional): Units for each input channel
        """
        
        # Ensure y is 2D (outputs x samples)
        if y is not None:
            y = np.atleast_1d(y)
            if y.ndim == 1:
                y = y.reshape(1, -1)  # Convert to 2D with 1 output
        
        # Ensure u is 2D (inputs x samples)
        if u is not None:
            u = np.atleast_1d(u)
            if u.ndim == 1:
                u = u.reshape(1, -1)  # Convert to 2D with 1 input
        
        self.y = y
        self.u = u
        self.samplingTime = samplingTime
        self.timestamps = np.array(timestamps)
        
        # Get dimensions
        self.n_outputs = y.shape[0]
        self.n_inputs = u.shape[0]
        self.n_samples = y.shape[1]
        
        # Set metadata with proper defaults if not provided
        self.y_names = list(y_names) if y_names is not None else [f"y{i+1}" for i in range(self.n_outputs)]
        self.u_names = list(u_names) if u_names is not None else [f"u{i+1}" for i in range(self.n_inputs)]
        self.y_units = list(y_units) if y_units is not None else ["" for _ in range(self.n_outputs)]
        self.u_units = list(u_units) if u_units is not None else ["" for _ in range(self.n_inputs)]
        
        # Ensure metadata lists match array dimensions, truncating or extending as necessary
        if len(self.y_names) != self.n_outputs:
            if len(self.y_names) > self.n_outputs:
                # Truncate if too many names
                self.y_names = self.y_names[:self.n_outputs]
            else:
                # Extend with default names if too few
                self.y_names.extend([f"y{i+1}" for i in range(len(self.y_names), self.n_outputs)])
                
        if len(self.u_names) != self.n_inputs:
            if len(self.u_names) > self.n_inputs:
                self.u_names = self.u_names[:self.n_inputs]
            else:
                self.u_names.extend([f"u{i+1}" for i in range(len(self.u_names), self.n_inputs)])
                
        if len(self.y_units) != self.n_outputs:
            if len(self.y_units) > self.n_outputs:
                self.y_units = self.y_units[:self.n_outputs]
            else:
                self.y_units.extend(["" for _ in range(len(self.y_units), self.n_outputs)])
                
        if len(self.u_units) != self.n_inputs:
            if len(self.u_units) > self.n_inputs:
                self.u_units = self.u_units[:self.n_inputs]
            else:
                self.u_units.extend(["" for _ in range(len(self.u_units), self.n_inputs)])
        
        # Normalization attributes
        self.y_norm_params = None
        self.u_norm_params = None
        self.is_normalized = False