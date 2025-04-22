import numpy as np
import os

class StateSpace:
    
    def __init__(self, A, B, C, D, K=None, x0=None, samplingTime=0):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.K = K
        self.x0 = x0
    
        self.samplingTime = samplingTime  # 0 = continuous time, otherwise discrete
    
    
    
    def save(self, filename):
        """
        Save the state-space model to a file using NumPy's .npz format.
        
        Parameters:
        -----------
        filename : str
            The filename to save to. If no extension is provided, .npz will be added.
        """
        # Add .npz extension if not already present
        if not filename.endswith('.npz'):
            filename = filename + '.npz'
        
        # Convert all attributes to NumPy arrays or basic Python types
        save_dict = {
            'A': self.A,
            'B': self.B,
            'C': self.C,
            'D': self.D,
            'samplingTime': self.samplingTime
        }
        
        # Save K and x0 only if they exist
        if self.K is not None:
            save_dict['K'] = self.K
        
        if self.x0 is not None:
            save_dict['x0'] = self.x0
        
        # Save to file
        np.savez(filename, **save_dict)
        print(f"Model saved to {filename}")
    
    @classmethod
    def load(cls, filename):
        """
        Load a state-space model from a .npz file.
        
        Parameters:
        -----------
        filename : str
            The filename to load from.
            
        Returns:
        --------
        StateSpace
            The loaded state-space model.
        """
        # Add .npz extension if not already present
        if not filename.endswith('.npz'):
            filename = filename + '.npz'
        
        # Load the data
        data = np.load(filename, allow_pickle=True)
        
        # Extract model parameters
        A = data['A']
        B = data['B'] 
        C = data['C']
        D = data['D']
        samplingTime = float(data['samplingTime'])
        
        # Handle optional parameters
        K = data['K'] if 'K' in data else None
        x0 = data['x0'] if 'x0' in data else None
        
        # Create and return the model
        model = cls(A, B, C, D, K, x0, samplingTime)
        print(f"Model loaded from {filename}")
        return model


# Example usage:
if __name__ == "__main__":
    # Define a simple model
    A = np.array([[0.8, 0.1], [0, 0.9]])
    B = np.array([[0.5], [1.0]])
    C = np.array([[1.0, 0.2]])
    D = np.array([[0]])
    K = np.array([[0.1, 0.2]])
    
    # Create model
    model = StateSpace(A, B, C, D, K=K, samplingTime=0.1)
    
    # Save model
    model.save("my_model")
    
    # Load model
    loaded_model = StateSpace.load("my_model")
    
    # Verify it loaded correctly
    print("\nOriginal model:")
    print(f"A: {model.A}")
    print(f"K: {model.K}")
    print(f"samplingTime: {model.samplingTime}")
    
    print("\nLoaded model:")
    print(f"A: {loaded_model.A}")
    print(f"K: {loaded_model.K}")
    print(f"samplingTime: {loaded_model.samplingTime}")