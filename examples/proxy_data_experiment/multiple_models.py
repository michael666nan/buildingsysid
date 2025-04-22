"""
=======================================================================
Example: Train Multiple Models and Compare
=======================================================================

Description:
This example demonstrates the following workflow:
  - ADD DESCRIPTION

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
output_data = room_temp.flatten()
output_names = ["Indoor Temp"]
output_units= ["°C"]

# input
input_data = np.stack([outdoor_temp, solar, proxy], axis=0)
input_names = ["Outdoor Temp", "Solar Gain", "Heat Proxy"]
u_units=["°C", "W/m2", "-"]

data = bid.IDData(
       y=output_data, 
       u=input_data,
       y_names=output_names,        # optional - used for plotting
       u_names=input_names,         # optional - used for plotting
       y_units=output_units,        # optional - used for plotting
       u_units=u_units,             # optional - used for plotting
       samplingTime=60*60,          # sampling time (seconds)
       timestamps=timestamps
       )  


# =====================================================================
# Step 3: Split and Plot Data for Training and Validation
# =====================================================================
train, val = data.split(train_ratio=0.7)  # 70% for training, 30% for validation


# =================================================================
# Step 4: Define Model Structures
# =================================================================

### First order model ###
fixed_params = {
    "ws1": 0.0,
    "wh1": 5.0
    }

param_bounds = {
    "x1[0]": [15, 25]
    }

grey1 = bid.model_set.grey.First(
            fixed_params=fixed_params, 
            param_bounds=param_bounds
            )

### Second order model ###
fixed_params = {
    "ws1": 0.0,
    "ws2": 0.0,
    "wh1": 5.0,
    "wh2": 0.0,
    }

param_bounds = {
    "x1[0]": [15, 30],
    "x2[0]": [15, 30]
    }

grey2 = bid.model_set.grey.Second(
            fixed_params=fixed_params, 
            param_bounds=param_bounds
            )

### Third order model ###
fixed_params = {
    "ws1": 0.0,
    "ws2": 0.0,
    "ws3": 0.0,
    "wh1": 5.0,
    "wh2": 0.0,
    "wh3": 0.0
    }

param_bounds = {
    "x1[0]": [15, 30],
    "x2[0]": [15, 30],
    "x3[0]": [15, 30]
    }

grey3 = bid.model_set.grey.Third(
            fixed_params=fixed_params, 
            param_bounds=param_bounds
            )


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
opt_problem1 = bid.calculate.OptimizationManager(
    model_structure=grey1,             # model structure
    data=train,                         # training data
    objective=multistep,                # criterion of fit
    solver=ls_solver                    # solver
) 

opt_problem2 = bid.calculate.OptimizationManager(
    model_structure=grey2,             # model structure
    data=train,                         # training data
    objective=multistep,                # criterion of fit
    solver=ls_solver                    # solver
) 

opt_problem3 = bid.calculate.OptimizationManager(
    model_structure=grey3,             # model structure
    data=train,                         # training data
    objective=multistep,                # criterion of fit
    solver=ls_solver                    # solver
)          


# =================================================================
# Step 9: Solve Optimization Problem
# =================================================================
grey1_opt = opt_problem1.solve(  
    max_attempts = 10,                          # maximum attemps to find a solution
    max_rmse=2,                               # maximum acceptable cost
    initialization_strategies=["grey_box"]      # strategy to create initial guess
)

grey2_opt = opt_problem2.solve(
    max_attempts = 10,                          # maximum attemps to find a solution
    max_rmse=grey1_opt.result.rmse,             # maximum acceptable cost
    initialization_strategies=["grey_box"]      # strategy to create initial guess
)

grey3_opt = opt_problem3.solve(
    max_attempts = 10,                          # maximum attemps to find a solution
    max_rmse=grey2_opt.result.rmse,             # maximum acceptable cost
    initialization_strategies=["grey_box"]      # strategy to create initial guess
)


# =================================================================
# Step 9: Print Estimated Parameters and Confidence Intervals
# =================================================================
grey1_opt.print()
grey2_opt.print()
grey3_opt.print()


# =================================================================
# Step 9: Compare model predictions with Validation Data
# =================================================================
# get state space model from estimated model structure
ss1 = grey1_opt.get_state_space()
ss2 = grey2_opt.get_state_space()
ss3 = grey3_opt.get_state_space()

list_of_models = [ss1, ss2, ss3]
list_of_names = ["ss1", "ss2", "ss3"]

# one-step predictions:                                                            
fit, y_sim = bid.validation.compare(
            list_of_models,                 # models
            val,                            # data
            model_names=list_of_names,    # list of names
            kstep=1,                        # prediction horizon
            title="one-step predictions")   # title on plot (optional)

# 12-step predictions:                                                              
_, _ = bid.validation.compare(
            list_of_models,                 # models
            val,                            # data
            model_names=list_of_names,    # list of names
            kstep=12,                       # prediction horizon
            title="12-step predictions")   # title on plot (optional)

# simulation (inf-step predictions):                                                              
_, _ = bid.validation.compare(
            list_of_models,                 # models
            val,                            # data
            model_names=list_of_names,    # list of names
            title="simulation")   # title on plot (optional)



# =================================================================
# Step 9: Save Model for later use
# =================================================================
#ss1.save(r"my_models\first_order_grey")  
