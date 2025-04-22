class IDDataSplit:
    """Mixin with resampling functionality"""

    def split(self, train_ratio=0.7):
        """
        Split the time series data into training and validation sets.
        
        For time series data, this maintains the temporal order by using
        the earlier portion for training and the later portion for validation.
        
        Args:
            train_ratio (float): Proportion of data to use for training (0 to 1)
            
        Returns:
            tuple: (train_data, val_data) - Two IDData objects containing the split data
        """
        # Determine split index
        n_samples = len(self.timestamps)
        split_idx = int(n_samples * train_ratio)
        
        # Split the data - arrays are guaranteed to be 2D at this point
        y_train = self.y[:, :split_idx]
        y_val = self.y[:, split_idx:]
        u_train = self.u[:, :split_idx]
        u_val = self.u[:, split_idx:]

        
        # Explicitly pass ALL metadata to the new objects
        train_data = self.__class__(
            y=y_train,
            u=u_train,
            samplingTime=self.samplingTime,
            timestamps=self.timestamps[:split_idx],
            y_names=list(self.y_names),  # Create new copies using list()
            u_names=list(self.u_names),
            y_units=list(self.y_units),
            u_units=list(self.u_units)
        )
        
        val_data = self.__class__(
            y=y_val,
            u=u_val,
            samplingTime=self.samplingTime,
            timestamps=self.timestamps[split_idx:],
            y_names=list(self.y_names),
            u_names=list(self.u_names),
            y_units=list(self.y_units),
            u_units=list(self.u_units)
        )
        
        # Transfer normalization information if present
        if self.is_normalized:
            train_data.y_norm_params = self.y_norm_params
            train_data.u_norm_params = self.u_norm_params
            train_data.is_normalized = True
            
            val_data.y_norm_params = self.y_norm_params
            val_data.u_norm_params = self.u_norm_params
            val_data.is_normalized = True
        
        return train_data, val_data