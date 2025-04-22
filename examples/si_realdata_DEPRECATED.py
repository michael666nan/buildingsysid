"""
=======================================================================
Example: System Identification using Real Data
=======================================================================

Description:
This example demonstrates the complete system identification workflow:
  - Preprocessing and analyzing simulation data
  - Estimating multiple grey-box thermal models
  - Validating models through residual analysis

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
       samplingTime=60*15,          # sampling time (seconds)
       timestamps=timestamps
       )  


# =====================================================================
# Step 8: Split and Plot Data for Training and Validation
# =====================================================================
# split
train, val = data.split(train_ratio=0.7)  # 70% for training, 30% for validation

# plot
train.plot_timeseries()
val.plot_timeseries()


# =================================================================
# Step 9: Define Model Set
# =================================================================
floor_area = 11

# first order models
#model1 = grey.First(floor_area=floor_area)
grey2 = bid.grey.Second(floor_area=floor_area)
# model3 = grey.Third(floor_area=floor_area)

black2 = bid.black.Second()

# =================================================================
# Step 10: Estimate Grey-Box Models
# =================================================================

grey2_sim = bid.pem(grey2, train)             # Objective: Simulation Errors
grey2_pred = bid.pem(grey2, train, kstep=1)   # Objective: One-Step Prediction Errors

black2_sim = bid.pem(black2, train)             # Objective: Simulation Errors
black2_pred = bid.pem(black2, train, kstep=1)   # Objective: One-Step Prediction Errors

# =================================================================
# Step 11: Compare
# =================================================================
# List state space models to compare
list_of_models = [grey2_sim.ss, grey2_pred.ss, black2_sim.ss, black2_pred.ss]
list_of_names = ["2 order grey (sim)", "2 order grey (one-step)", 
                 "2 order black (sim)", "2 order black (one-step)"]

# Plot model performance on validation data
bid.compare(list_of_models, train, model_names=list_of_names, title="Validation (One-Step)", kstep=1)

bid.compare(list_of_models, train, model_names=list_of_names, title="Validation (Simulation)")


# # # # # =================================================================
# # # # # Step 12: Residuals Analysis
# # # # # =================================================================
# # # from validation.residual_analysis import perform_residual_analysis

# # # _ = perform_residual_analysis(model3_est.residuals, inputs=all_data.u)

