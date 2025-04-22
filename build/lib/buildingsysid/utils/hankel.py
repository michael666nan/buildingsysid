import numpy as np

def hankel(kstep, vector):
        """
        Create a Hankel matrix from a vector with a specified number of rows.
        
        Parameters:
            kstep (int): The number of rows in the Hankel matrix.
            vector (array-like): List or array containing the elements of the vector.
        
        Returns:
            numpy.ndarray: The resulting Hankel matrix.
        """
        
        # Ensure vector is 1D
        vector = np.ravel(vector)
        
        # Determine the number of columns
        num_cols = len(vector) - kstep + 1
        
        # Create an empty matrix with the specified number of rows and calculated number of columns
        hankel_matrix = np.zeros((kstep, num_cols))
        
        # Fill the Hankel matrix using the vector
        for i in range(kstep):
            for j in range(num_cols):
                hankel_matrix[i, j] = vector[i + j]
                
        return hankel_matrix
