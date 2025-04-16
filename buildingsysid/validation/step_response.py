import numpy as np
import matplotlib.pyplot as plt

def step_response(ss, iddata, input_channels=None, N=72, plot=True, output_names=None, title=None):
    """
    Calculate and optionally plot the step response of a state-space model.
    
    Parameters:
    -----------
    ss : object
        State-space model object containing A, B, C, D matrices
    iddata : object
        Data object (not used for calculation but kept for compatibility)
    input_channels : list or None, optional
        List of input channels to use for the step response (default: None, meaning all inputs)
    N : int, optional
        Number of time steps to simulate (default: 72)
    plot : bool, optional
        Whether to plot the step response (default: True)
    output_names : list, optional
        Names of the output channels for the plot legend (default: None)
    title : str, optional
        Custom title for the plot (default: None)
        
    Returns:
    --------
    y_sim : dict
        Dictionary of simulated outputs for each input channel
    x_sim : dict
        Dictionary of state trajectories for each input channel
    fig : matplotlib.figure.Figure or None
        Figure object if plot=True, otherwise None
    ss_gains : dict
        Dictionary of steady-state gains for each input channel
    """
    
    A = ss.A 
    B = ss.B 
    C = ss.C 
    D = ss.D 
    
    # Determine which input channels to use
    if input_channels is None:
        input_channels = list(range(B.shape[1]))
    elif isinstance(input_channels, int):
        input_channels = [input_channels]
    
    # Create dictionaries to store results for each input channel
    y_sim = {}
    x_sim = {}
    ss_gains = {}
    
    # Calculate steady-state gains
    try:
        # Computing (I-A)^(-1) for steady-state calculation
        I = np.eye(A.shape[0])
        I_minus_A_inv = np.linalg.solve(I - A, I)  # More stable than direct inverse
    except np.linalg.LinAlgError:
        print("Warning: Matrix (I-A) is singular. Steady-state gains may not be accurate.")
        I_minus_A_inv = None
    
    # Simulate step response for each input channel
    for input_channel in input_channels:
        # Initialize
        y = np.zeros((C.shape[0], N))
        x = np.zeros((A.shape[0], N))
        y[:,0] = C @ x[:,0] + D[:,input_channel] * 0  # No input at t=0
        
        # Simulate system dynamics
        for k in range(N-1):        
            x[:,k+1] = A @ x[:,k] + B[:,input_channel]
            y[:,k+1] = C @ x[:,k+1] + D[:,input_channel]
        
        # Store results
        y_sim[input_channel] = y
        x_sim[input_channel] = x
        
        # Calculate steady-state gains
        if I_minus_A_inv is not None:
            steady_state_term = I_minus_A_inv @ B[:,input_channel:input_channel+1]
            ss_gain = C @ steady_state_term + D[:,input_channel:input_channel+1]
            ss_gains[input_channel] = ss_gain
        else:
            # Estimate steady-state gain from the final values
            ss_gains[input_channel] = y[:,-1] - y[:,0]
    
    # Create plot if requested
    fig = None
    if plot:
        num_inputs = len(input_channels)
        num_outputs = C.shape[0]
        
        # Always create separate subplots for each input channel
        fig, axes = plt.subplots(num_inputs, 1, figsize=(12, 4 * num_inputs), squeeze=False)
        
        for i, input_channel in enumerate(input_channels):
            ax = axes[i, 0]
            
            for j in range(num_outputs):
                if output_names is not None and j < len(output_names):
                    label = output_names[j]
                else:
                    label = f'Output {j+1}'
                
                response = y_sim[input_channel][j, :] - y_sim[input_channel][j, 0]  # Remove initial offset
                ax.plot(np.arange(N), response, label=label)
                
                # Plot dotted line at steady-state value
                ss_value = ss_gains[input_channel][j, 0] if isinstance(ss_gains[input_channel], np.ndarray) else ss_gains[input_channel][j]
                ax.axhline(y=ss_value, color=f'C{j}', linestyle='--', alpha=0.7)
            
            ax.set_title(f'Step Response - Input Channel {input_channel+1}' if title is None else f'{title} - Input {input_channel+1}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
    
    return y_sim, x_sim, fig, ss_gains