"""
=======================================================================
Example: Black-Box System Identification
=======================================================================

Description:
This example demonstrates the following workflow:
  - Define dataset
          - Load and preprocess
          - Put into IDData format
          - Split data into traing and validation data
  - Define black box model structure (linear time-invariant state-space)
  - Define criterion of fit (objective)
  - Define solver
  - Create optimization manager and find optimal parameters
  - Inspect estimated model
          - Confidence intervals
          - Compare against validation data
          - Residual analysis
  - Save model for later use

=======================================================================
"""

import pandas as pd
import numpy as np
import buildingsysid as bid


# =====================================================================
# Step 1: Load data and Prepare Data
# =====================================================================
# Load the data
df = pd.read_csv('preprocessed_building_data.csv', index_col='datetime', parse_dates=True)

# Drop rows with NaN
columns_of_interest = ["Aqara_room", "temperature_2m (°C)", "global_tilted_irradiance (W/m²)", 
                       "Aqara_supply", "Aqara_return"]
clean_df = df[columns_of_interest].dropna()

# Extract the data
room_temp = clean_df["Aqara_room"].to_numpy()
outdoor_temp = clean_df["temperature_2m (°C)"].to_numpy()
solar = clean_df["global_tilted_irradiance (W/m²)"].to_numpy()
supply_temp = clean_df["Aqara_supply"].to_numpy()
return_temp = clean_df["Aqara_return"].to_numpy()
timestamps = clean_df.index

# Calculate proxy
proxy = ((supply_temp+return_temp)/2 - room_temp)
proxy = np.maximum(proxy, 0)
proxy = proxy**1.1


# =====================================================================
# Step 2: Create IDData
# =====================================================================
# output
output_data = room_temp.flatten() + 273.15
output_names = ["Indoor Temp"]
output_units= ["°C"]

# input
input_data = np.stack([outdoor_temp+273.15, solar, proxy], axis=0)
input_names = ["Outdoor Temp", "Solar Gain", "Heat Proxy"]
u_units=["°C", "W/m2", "-"]

data = bid.IDData(
       y=output_data, 
       u=input_data,
       y_names=output_names,        # optional - used for plotting
       u_names=input_names,         # optional - used for plotting
       y_units=output_units,        # optional - used for plotting
       u_units=u_units,             # optional - used for plotting
       samplingTime=60*15,          # sampling time (seconds)
       timestamps=timestamps
       )  


# =====================================================================
# Step 3: Split and Plot Data for Training and Validation
# =====================================================================
train, val = data.split(train_ratio=0.7)  # 70% for training, 30% for validation


# =================================================================
# Step 4: Define Model Structure
# =================================================================
fixed_params = {
    "b12": 0,
    "b22": 0
}
black2 = bid.model_set.black.Second()


# =================================================================
# Step 9: Define Criterion of Fit
# =================================================================
multistep = bid.criterion_of_fit.StandardObjective(kstep=1, sum_hor=False)


# =================================================================
# Step 9: Define Solver
# =================================================================
ls_solver = bid.calculate.LeastSquaresSolver(
    method='trf',
    verbose=2,
    ftol=1e-6
)


# =================================================================
# Step 9: Construct Optimization Problem
# =================================================================
opt_problem = bid.calculate.OptimizationManager(
    model_structure=black2,             # model structure
    data=train,                         # training data
    objective=multistep,                # criterion of fit
    solver=ls_solver                    # solver
)    


# =================================================================
# Step 9: Solve Optimization Problem
# =================================================================
black2_opt = opt_problem.solve(                     # maximum attemps to find a solution
    initialization_strategies=["biased_random"]     # strategy to create initial guess
)


# =================================================================
# Step 9: Print Estimated Parameters and Confidence Intervals
# =================================================================
black2_opt.print()


# =================================================================
# Step 9: Compare model predictions with Validation Data
# =================================================================
# get state space model from estimated model structure
ss2 = black2_opt.get_state_space()

# one-step predictions:                                                            
fit, y_sim = bid.validation.compare(
            ss2,                            # model
            val,                            # data
            kstep=1,                       # prediction horizon
            title="one-step predictions")   # title on plot (optional)

# 12-step predictions:                                                              
_, _ = bid.validation.compare(
            ss2,                            # model
            val,                            # data
            kstep=12,                       # prediction horizon
            title="12-step predictions")   # title on plot (optional)

# simulation (inf-step predictions):                                                              
_, _ = bid.validation.compare(
            ss2,                            # model
            val,                            # data
            title="simulation")   # title on plot (optional)


# =================================================================
# Step 9: Residual Analysis
# =================================================================
# calculate residuals
residuals = val.y - y_sim
bid.validation.perform_residual_analysis(residuals)


# =================================================================
# Step 9: Save Model for later use
# =================================================================
ss2.save("my_models\second_order")                                                             
                                                                


