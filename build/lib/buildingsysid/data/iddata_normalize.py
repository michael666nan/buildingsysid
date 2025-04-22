import numpy as np
import copy

class IDDataNormalize:
    """Mixin with normalization functionality for data and models"""
    
    def normalize(self):
        """
        Normalize input and output data by dividing each channel by its maximum value.
        Returns a new IDData object with normalized data and stores normalization factors.
        
        Returns:
        --------
        IDData
            A new IDData object with normalized data
        """
        # Create copies to avoid modifying original data
        u_norm = self.u.copy()
        y_norm = self.y.copy()
        
        # Store maximum values for each channel
        u_max = np.zeros(self.n_inputs)
        y_max = np.zeros(self.n_outputs)
        
        # Normalize each input channel
        for i in range(self.n_inputs):
            u_max[i] = np.max(np.abs(self.u[i,:]))
            if u_max[i] > 0:  # Avoid division by zero
                u_norm[i,:] = self.u[i,:] / u_max[i]
        
        # Normalize each output channel
        for i in range(self.n_outputs):
            y_max[i] = np.max(np.abs(self.y[i,:]))
            if y_max[i] > 0:  # Avoid division by zero
                y_norm[i,:] = self.y[i,:] / y_max[i]
        
        # Create a new IDData object with normalized data
        normalized_data = self.__class__(
            y=y_norm,
            u=u_norm,
            samplingTime=self.samplingTime,
            timestamps=self.timestamps,
            y_names=self.y_names.copy(),
            u_names=self.u_names.copy(),
            y_units=self.y_units.copy(),
            u_units=self.u_units.copy()
        )
        
        # Store normalization factors in the new object
        normalized_data.u_max = u_max
        normalized_data.y_max = y_max
        normalized_data.is_normalized = True
        normalized_data.original_data = self  # Reference to original data
        
        return normalized_data
    
    def denormalize(self):
        """
        Convert normalized data back to original scale.
        Only works if the object was created through the normalize() method.
        
        Returns:
        --------
        IDData
            A new IDData object with denormalized data
        """
        if not hasattr(self, 'is_normalized') or not self.is_normalized:
            print("Warning: Data is not normalized, returning a copy of the original data")
            return copy.deepcopy(self)
        
        if not hasattr(self, 'u_max') or not hasattr(self, 'y_max'):
            print("Warning: Normalization factors not found, returning a copy of the original data")
            return copy.deepcopy(self)
        
        # Create copies of normalized data
        u_denorm = self.u.copy()
        y_denorm = self.y.copy()
        
        # Denormalize each input channel
        for i in range(self.n_inputs):
            if self.u_max[i] > 0:  # Avoid multiplication by zero
                u_denorm[i,:] = self.u[i,:] * self.u_max[i]
        
        # Denormalize each output channel
        for i in range(self.n_outputs):
            if self.y_max[i] > 0:  # Avoid multiplication by zero
                y_denorm[i,:] = self.y[i,:] * self.y_max[i]
        
        # Create a new IDData object with denormalized data
        denormalized_data = self.__class__(
            y=y_denorm,
            u=u_denorm,
            samplingTime=self.samplingTime,
            timestamps=self.timestamps,
            y_names=self.y_names.copy(),
            u_names=self.u_names.copy(),
            y_units=self.y_units.copy(),
            u_units=self.u_units.copy()
        )
        
        return denormalized_data
    
    def denormalize_state_space(self, ss):
        """
        Transform a normalized state-space model to use original unscaled inputs and outputs.
        
        Parameters:
        -----------
        ss : object
            Normalized state-space model with A, B, C, D attributes
            
        Returns:
        --------
        ss_denorm : object
            Denormalized state-space model
        """
        if not hasattr(self, 'is_normalized') or not self.is_normalized:
            print("Warning: Data is not normalized, state-space model will not be transformed")
            return copy.deepcopy(ss)
        
        if not hasattr(self, 'u_max') or not hasattr(self, 'y_max'):
            print("Warning: Normalization factors not found, state-space model will not be transformed")
            return copy.deepcopy(ss)
        
        # Get normalization factors
        u_max = self.u_max
        y_max = self.y_max
        
        # Apply static method with proper scaling factors
        return self.denormalize_state_space_static(ss, u_max, y_max)
    
    @staticmethod
    def denormalize_state_space_static(ss, u_max, y_max):
        """
        Static method to transform a normalized state-space model using provided scaling factors.
        
        For a system identified with normalized data:
        - ẋ = A·x + B_norm·u_norm + K_norm·(y_norm - C_norm·x - D_norm·u_norm)
        - y_norm = C_norm·x + D_norm·u_norm
        
        Where:
        - u_norm = u / u_max
        - y_norm = y / y_max
        
        The denormalized matrices are:
        - B = B_norm / u_max (element-wise)
        - C = C_norm * y_max (element-wise)
        - D = D_norm * y_max / u_max (element-wise)
        - K = K_norm / y_max (element-wise) - for Kalman gain
        
        Parameters:
        -----------
        ss : object
            Normalized state-space model with A, B, C, D attributes
            and optionally K (Kalman gain) attribute
        u_max : ndarray
            Maximum values used to normalize inputs
        y_max : ndarray
            Maximum values used to normalize outputs
            
        Returns:
        --------
        ss_denorm : object
            Denormalized state-space model
        """
        # Get normalized state-space matrices
        A = ss.A
        B_norm = ss.B
        C_norm = ss.C
        D_norm = ss.D
        
        # Check if Kalman gain exists
        has_kalman_gain = hasattr(ss, 'K') and ss.K is not None
        if has_kalman_gain:
            K_norm = ss.K
        
        # Create scaling matrices
        n_inputs = len(u_max)
        n_outputs = len(y_max)
        
        # Element-wise division for B
        B_denorm = np.zeros_like(B_norm)
        for i in range(B_norm.shape[1]):
            if i < n_inputs and u_max[i] > 0:
                B_denorm[:, i] = B_norm[:, i] / u_max[i]
        
        # Element-wise multiplication for C
        C_denorm = np.zeros_like(C_norm)
        for i in range(C_norm.shape[0]):
            if i < n_outputs and y_max[i] > 0:
                C_denorm[i, :] = C_norm[i, :] * y_max[i]
        
        # Element-wise operations for D
        D_denorm = np.zeros_like(D_norm)
        for i in range(D_norm.shape[0]):
            for j in range(D_norm.shape[1]):
                if i < n_outputs and j < n_inputs and u_max[j] > 0:
                    D_denorm[i, j] = D_norm[i, j] * y_max[i] / u_max[j]
        
        # Handle Kalman gain matrix (K) if it exists
        if has_kalman_gain:
            K_denorm = np.zeros_like(K_norm)
            for i in range(K_norm.shape[1]):
                if i < n_outputs and y_max[i] > 0:
                    # Kalman gain needs to be divided by y_max because
                    # it multiplies the output error (y - ŷ) which is normalized
                    K_denorm[:, i] = K_norm[:, i] / y_max[i]
        
        # Create new state-space model
        ss_denorm = copy.deepcopy(ss)
        ss_denorm.A = A.copy()  # A remains unchanged
        ss_denorm.B = B_denorm
        ss_denorm.C = C_denorm
        ss_denorm.D = D_denorm
        
        # Set the transformed Kalman gain
        if has_kalman_gain:
            ss_denorm.K = K_denorm
        
        return ss_denorm