import matplotlib.pyplot as plt
import numpy as np
import copy
from buildingsysid.criterion_of_fit.simulate_statespace import simulation


def compare(models, iddata, kstep=-1, sum_horizon=False, initial_state="estimate", 
            model_names=None, colors=None, title=None):
    """
    Compare one or multiple state-space models against identification data.
    
    Args:
        models: Single state-space model or list of models to compare
        iddata: Identification data object containing inputs and outputs
        kstep: Prediction horizon for simulation:
            kstep < 0 --> Simulation
            kstep = 1 --> one step ahead predictions
            kstep > 1 --> k step ahead predictions
        sum_horizon: Whether to summarize over the horizon
        initial_state: "estimate" to estimate initial states or "zero" for zero initial states
        model_names: Optional list of names for each model (default: "Model 1", "Model 2", etc.)
        colors: Optional list of colors for plotting each model
        
    Returns:
        fits: List of fit percentages for each model (or single value if one model)
        y_sims: List of simulated outputs for each model (or single array if one model)
        fig: The generated figure
    """
    
    # Convert single model to list for consistent handling
    single_model = False
    if not isinstance(models, list):
        models = [models]
        single_model = True
    
    # Create default model names if not provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    elif len(model_names) < len(models):
        # Extend with default names if not enough names provided
        model_names.extend([f"Model {i+1}" for i in range(len(model_names), len(models))])
    
    # Set default colors if not provided
    if colors is None:
        # Define a color cycle - can be extended with more colors
        default_colors = ['r', 'g', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = [default_colors[i % len(default_colors)] for i in range(len(models))]
    elif len(colors) < len(models):
        # Extend with default colors if not enough colors provided
        default_colors = ['r', 'g', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors.extend([default_colors[i % len(default_colors)] for i in range(len(colors), len(models))])
    
    # Calculate measured output transformation for fit calculation
    if sum_horizon:
        Y = iddata.hankel(kstep)
        y_mea = Y.T.flatten().reshape((1,-1))
    else:
        y_mea = iddata.y
    
    # Lists to store results
    fits = []
    y_sims = []
    
    # Process each model
    for i, model in enumerate(models):
        # Create a deep copy to avoid modifying the original
        ss = copy.deepcopy(model)
        
        # Estimate initial state if requested
        if initial_state == "estimate":
            ss.x0 = estimate_initial_state(ss, iddata, kstep=1)
        
        # Simulate the model
        y_sim = simulation(ss, iddata, kstep=kstep, sum_horizon=sum_horizon)
        y_sims.append(y_sim)
        
        # Calculate fit
        residuals = y_mea[0,:] - y_sim[0,:]
        num = np.sqrt(np.sum(residuals**2))
        den = np.sqrt(np.sum((y_mea[0,:] - np.mean(y_mea[0,:]))**2))
        fit = 100 * (1 - num / den)
        fits.append(fit)
    
    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    markersize = 3
    
    # Subplot 1: Measured vs. Predicted Indoor Temperature for all models
    axes[0].plot(iddata.timestamps, iddata.y[0, :], label="Measured Temperature", 
                 marker=".", markersize=markersize, color='b')
    
    # Plot each model's prediction
    for i, y_sim in enumerate(y_sims):
        axes[0].plot(iddata.timestamps, y_sim[0, :], 
                     label=f"{model_names[i]} ({fits[i]:.2f} %)", 
                     marker=".", markersize=markersize, linestyle='dashed', 
                     color=colors[i])
    
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(f"Measured vs. Predicted Indoor Temperature {(title)}")
    axes[0].legend()
    axes[0].grid(True)
    
    # Subplot 2: Solar Radiation & Radiator Heat Input (Dual Y-Axis)
    ax2 = axes[1].twinx()
    axes[1].step(iddata.timestamps, iddata.u[1, :], label="Solar Radiation", 
                 where="post", marker=".", markersize=markersize, color='orange')
    ax2.step(iddata.timestamps, iddata.u[2, :], label="Radiator Heat Input", 
              where="post", marker=".", markersize=markersize, color='purple')
    axes[1].set_ylabel("Solar Radiation (W/m²)", color='orange')
    ax2.set_ylabel("Radiator Heat Input (W)", color='purple')
    axes[1].set_title("Solar Radiation & Radiator Heat Input")
    axes[1].grid(True)
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Subplot 3: Outdoor Temperature
    axes[2].step(iddata.timestamps, iddata.u[0, :], label="Outdoor Temperature", 
                 where="post", marker=".", markersize=markersize, color='g')
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_title("Outdoor Temperature")
    axes[2].legend()
    axes[2].grid(True)
    
    # Formatting x-axis
    plt.xlabel("Time")
    
    plt.tight_layout()
    plt.show()
    
    # Return single values if only one model was provided
    if single_model:
        return fits[0], y_sims[0]
    else:
        return fits, y_sims


def estimate_initial_state(ss, iddata, kstep=1):
    """
    Estimate the initial state of a state-space model based on identification data.
    
    Args:
        ss: State-space model
        iddata: Identification data object
        kstep: Prediction horizon
        
    Returns:
        x0: Estimated initial state
    """
    from scipy.optimize import least_squares
    
    def objective(x0):
        ss.x0[:,0] = x0
        y_sim = simulation(ss, iddata, kstep, sum_horizon=False)
        error = iddata.y[0,:] - y_sim[0,:]
        return error.flatten()
    
    # Initial guess (zeros)
    x0_guess = ss.x0.flatten()
    
    # Run the optimization
    result = least_squares(objective, x0_guess)
    
    # Get the optimal initial state
    ss.x0[:,0] = result.x
    
    return ss.x0


# Example usage:
if __name__ == "__main__":
    # Example with multiple models
    # models = [model1, model2, model3]
    # model_names = ["H2C2", "H2C2Split", "Full2"]
    # fits, y_sims, fig = compare(models, iddata, model_names=model_names)
    
    # Example with a single model
    # fit, y_sim, fig = compare(model1, iddata)
    pass