"""
=======================================================================
Example: Black-Box System Identification for Building Thermal Modeling
=======================================================================

Description:
This example demonstrates how to use black-box system identification
to create a thermal model of a building from data. The workflow includes:
  - Loading and preprocessing building simulation data
  - Creating and resampling identification data
  - Splitting data into training and validation sets
  - Defining two general black-box model structures (first and second order)
  - Training the model using optimization
  - Validating model performance
  - Analyzing residuals and saving the model

Black-box models use a mathematical structure without requiring
physical knowledge of the system, making them versatile for many
applications.
=======================================================================
"""
import pandas as pd
import buildingsysid as bid


# =====================================================================
# Step 1: Load and Prepare Data
# =====================================================================
print("Step 1: Loading and preprocessing data...")
# Load data from EnergyPlus simulation results
df = pd.read_csv('eplus_example1.csv', index_col='time_stamp', parse_dates=True) 

# Extract timestamps for time series handling
timestamps = df.index

# Define output variable (what we want to predict)
output_name = ["Indoor Temp"]                
output_unit = ["°C"]                     
output_data = df[output_name].to_numpy().T                  # Transpose to get correct shape

# Define input variables (what drives the system)
input_names = ["Outdoor Temp", "Solar Gain", "Heat Power"]  
input_units = ["°C", "W/m²", "W"]                        
input_data = df[input_names].to_numpy().T                   # Transpose to get correct shape


# =====================================================================
# Step 2: Create Identification Data Object
# =====================================================================
print("Step 2: Creating identification data object...")
# Original data sampling time
sampling_time = 60   # 60 seconds (1 minute)

# Create the IDData object which holds all data for system identification
data = bid.IDData(
       y=output_data,              # Output data (indoor temperature)
       u=input_data,               # Input data (outdoor temp, solar, heating)
       y_names=output_name,        # Name of output variable
       u_names=input_names,        # Names of input variables
       y_units=output_unit,        # Unit of output variable
       u_units=input_units,        # Units of input variables
       samplingTime=sampling_time, # Sampling time in seconds
       timestamps=timestamps       # Time stamps for data points
)  


# =====================================================================
# Step 3: Resample Data to Hourly Intervals
# =====================================================================
print("Step 3: Resampling data to hourly intervals for modeling...")
# Define the new sampling time (3600 seconds = 1 hour)
model_sampling_time = 3600  # seconds

# Resample the data for modeling (reduces noise and computation time)
data_resampled = data.resample(
    new_sampling_time=model_sampling_time,
    output_agg="first",   # Take first output value in each interval
    input_agg="mean"      # Take mean of input values in each interval
)


# =====================================================================
# Step 4: Split Data into Training and Validation Sets
# =====================================================================
print("Step 4: Splitting data into training (70%) and validation (30%) sets...")
# Split data - 70% for training, 30% for validation
train, val = data_resampled.split(train_ratio=0.7)

# Plot training data time series to visualize the data
train.plot_timeseries(title="Training Data")

# Plot validation data time series
val.plot_timeseries(title="Validation Data")

# Plot correlation to check relationships between variables
train.plot_cross_correlation()     # Shows how inputs affect output
train.plot_partial_cross_correlation()  # Shows direct relationships


# =====================================================================
# Step 5: Define Model Structure
# =====================================================================
print("Step 5: Defining black-box model structure...")
# Create an unconstrained first-order LTI (Linear Time-Invariant) black-box model
black1 = bid.model_set.black.First()   # first-order means it will have 1 internal state

# Create an unconstrained second-order LTI (Linear Time-Invariant) black-box model
black2 = bid.model_set.black.Second()   # Second-order means it will have 2 internal states


# =====================================================================
# Step 6: Define Prediction Objective
# =====================================================================
print("Step 6: Defining prediction objective...")
# Use one-step-ahead prediction as objective function
# This means we want to predict the next time step given current data
onestep_pred = bid.criterion_of_fit.StandardObjective(
    kstep=1,       # One-step ahead prediction
)


# =====================================================================
# Step 7: Define Optimization Solver
# =====================================================================
print("Step 7: Setting up optimization solver...")
# Use a least-squares solver to find optimal model parameters
ls_solver = bid.calculate.LeastSquaresSolver(
    method='trf',    # Trust Region Reflective algorithm
    verbose=2,       # Show detailed output during optimization
    ftol=1e-6        # Function tolerance for convergence
)


# =====================================================================
# Step 8: Set Up Optimization Problem
# =====================================================================
print("Step 8: Setting up optimization problem...")
# Configure the optimization problem combining model, data, objective and solver
opt_problem1 = bid.calculate.OptimizationManager(
    model_structure=black1,    # The model structure to optimize
    data=train,                # Training data
    objective=onestep_pred,    # Prediction objective
    solver=ls_solver           # Optimization solver
)    

opt_problem2 = bid.calculate.OptimizationManager(
    model_structure=black2,    # The model structure to optimize
    data=train,                # Training data
    objective=onestep_pred,    # Prediction objective
    solver=ls_solver           # Optimization solver
)    


# =====================================================================
# Step 9: Solve Optimization Problem (Train the Model)
# =====================================================================
print("Step 9: Training the model...")
# Train the models by solving the optimization problem
black1_opt = opt_problem1.solve(
    initialization_strategies=["black_box"]  # Start with values typical for stable building systems
)

black2_opt = opt_problem2.solve(
    initialization_strategies=["black_box"]  # Start with values typical for stable building systems
)

# =====================================================================
# Step 10: Analyze Model Parameters
# =====================================================================
print("Step 10: Analyzing estimated model parameters...")
# Display estimated parameters and their confidence intervals
black1_opt.print()
black2_opt.print()


# =====================================================================
# Step 11: Validate Model on Test Data
# =====================================================================
print("Step 11: Validating model performance...")
# Create state-space models using the optimized parameters
ss1 = black1_opt.get_state_space()
ss2 = black2_opt.get_state_space()

# Create list with models
model_list = [ss1, ss2]
model_names = ["ss1", "ss2"]

# Evaluate one-step ahead prediction (short-term forecast)
fit, y_sim = bid.validation.compare(
    model_list,                 # The models
    val,                        # Validation data
    kstep=1,                    # One-step ahead prediction
    model_names=model_names,    # For plotting
    title="One-step Predictions"
)

# Evaluate 12-step ahead prediction (medium-term forecast, 12 hours)
_, _ = bid.validation.compare(
    model_list,                 # The models
    val,                        # Validation data
    kstep=12,                   # 12-step ahead prediction
    model_names=model_names,    
    title="12-step Predictions"
)

# Evaluate full simulation (long-term behavior)
_, _ = bid.validation.compare(
    model_list,                     # The model
    val,                            # Validation data
    model_names=model_names,
    title="Simulation"              # Infinite-step simulation
)


# =====================================================================
# Step 12: Residual Analysis (Optional)
# =====================================================================
print("Step 12: Performing residual analysis...")
# Calculate residuals (difference between model predictions and actual data)
residuals = val.y - y_sim

# Analyze residuals to check model adequacy
# Uncomment the line below to perform and visualize residual analysis
# bid.validation.perform_residual_analysis(residuals)


# =====================================================================
# Step 13: Save Model for Future Use
# =====================================================================
print("Step 13: Saving model for future use...")
# # Uncomment the line below to save models
# ss1.save("first_order")
# ss2.save("second_order")

print("\nBlack-box system identification completed successfully!")