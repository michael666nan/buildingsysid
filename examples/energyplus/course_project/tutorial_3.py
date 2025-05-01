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
df = pd.read_csv('heavyweight_room_prbs.csv', index_col='time_stamp', parse_dates=True) 

# Read the CSV file with time stamps
df = pd.read_csv('heavyweight_room_prbs.csv', index_col='time_stamp', parse_dates=True) 

# Extract a specific time period for the analysis
# For example, selecting data from January
start_date = '2023-01-01'
end_date = '2023-01-31'
df = df.loc[start_date:end_date]



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
model_sampling_time = 3600  # seconds

data_resampled = data.resample(
    new_sampling_time=model_sampling_time,
    output_agg="first",
    input_agg="mean"
)


# =====================================================================
# Step 4: Split Data into Training and Validation Sets
# =====================================================================
print("Step 4: Splitting data into training (70%) and validation (30%) sets...")
train, val = data_resampled.split(train_ratio=0.7)
train.plot_timeseries()


# =====================================================================
# Step 5: Define Model Structure
# =====================================================================
print("Step 5: Defining black-box model structure...")
second = bid.model_set.black.Second()
third = bid.model_set.black.Third()

# =====================================================================
# Step 6: Define Objectives
# =====================================================================
print("Step 6: Defining prediction objective...")
objective = bid.criterion_of_fit.StandardObjective(kstep=12, sum_hor=True)


# =====================================================================
# Step 7: Define Optimization Solver
# =====================================================================
print("Step 7: Setting up optimization solver...")
ls_solver = bid.calculate.LeastSquaresSolver(
    method='trf',    # Trust Region Reflective algorithm
    verbose=2,       # Show detailed output during optimization
    ftol=1e-6        # Function tolerance for convergence
)


# =====================================================================
# Step 8: Set Up Optimization Problem
# =====================================================================
print("Step 8: Setting up optimization problem...")
opt_problem = bid.calculate.OptimizationManager(
    model_structure=third,     # The model structure to optimize
    data=train,                 # Training data
    objective=objective,        # Prediction objective
    solver=ls_solver            # Optimization solver
)    

# opt_problem12 = bid.calculate.OptimizationManager(
#     model_structure=third,     # The model structure to optimize
#     data=train,                 # Training data
#     objective=obj_12step,        # Prediction objective
#     solver=ls_solver            # Optimization solver
# )  

# =====================================================================
# Step 9: Solve Optimization Problem (Train the Model)
# =====================================================================
print("Step 9: Training the model...")
solution = opt_problem.solve(
    initialization_strategies=["black_box"]
)

# opt_12 = opt_problem12.solve(
#     x0 = opt_3.result.x,
#     initialization_strategies=[]
# )



# =====================================================================
# Step 10: Analyze Model Parameters
# =====================================================================
print("Step 10: Analyzing estimated model parameters...")
solution.print()


# =====================================================================
# Step 11: Validate Model on Test Data
# =====================================================================
print("Step 11: Validating model performance...")
ss = solution.get_state_space()


# Evaluate on training data
train1, _ = bid.validation.compare(ss, train, kstep=1, make_plot=False)
train12, _ = bid.validation.compare(ss, train, kstep=12, make_plot=False)
print("Traing:")
print(f"one-step: {train1}")
print(f"12-step: {train12}")


# Evaluate different objectives on validation data
fit1, _ = bid.validation.compare(ss, val, kstep=1, make_plot=False)
fit12, _ = bid.validation.compare(ss, val, kstep=12, make_plot=False)
fit48, _ = bid.validation.compare(ss, val, kstep=48, make_plot=False)
fit48sum, _ = bid.validation.compare(ss, val, kstep=48, sum_horizon=True, make_plot=False)

print("Validation:")
print(f"one-step fit: {fit1}")
print(f"12-step fit: {fit12}")
print(f"48-step fit: {fit48}")
print(f"48-sum fit: {fit48sum}")

# =====================================================================
# Step 13: Save Model for Future Use
# =====================================================================
# print("Step 13: Saving model for future use...")
ss.save(r"models\third_12sum")
