# Grey-Box System Identification for Building Thermal Modeling

This tutorial demonstrates a complete workflow for building thermal modeling using grey-box system identification techniques with the `buildingsysid` package.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Workflow Overview](#workflow-overview)
- [Step 1: Data Acquisition and Preprocessing](#step-1-data-acquisition-and-preprocessing)
- [Step 2: Creating IDData Objects for System Identification](#step-2-creating-iddata-objects-for-system-identification)
- [Step 3: Splitting Data](#step-3-splitting-data)
- [Step 4: Defining Grey-Box Model Structures](#step-4-defining-grey-box-model-structures)
- [Step 5: Setting Up Optimization](#step-5-setting-up-optimization)
- [Step 6: Training the Model](#step-6-training-the-model)
- [Step 7: Evaluating Model Performance](#step-7-evaluating-model-performance)
- [Step 8: Results and Analysis](#step-8-results-and-analysis)
- [Extending the Tutorial](#extending-the-tutorial)
- [References](#references)

## Introduction

Grey-box modeling is a powerful approach for building thermal modeling that combines physical knowledge with data-driven techniques. This approach is particularly useful for building energy management systems, HVAC control, and thermal comfort prediction.

In this tutorial, we'll use the `buildingsysid` package to create a thermal model of a building zone based on temperature and energy data. The resulting model can predict indoor temperature based on outdoor conditions and heating inputs.

## Prerequisites

To follow this tutorial, you'll need:

- Python 3.7+
- The following packages:
  - `buildingsysid`
  - `pandas`
  - `numpy`
  - `matplotlib` (for visualization)
  - `requests` (for downloading data)

You can install these packages using pip:

```bash
pip install buildingsysid pandas numpy matplotlib requests
```

## Workflow Overview

The system identification workflow consists of the following steps:

1. **Data acquisition and preprocessing**: Obtain and clean the data
2. **Data formatting**: Create IDData objects for system identification
3. **Data splitting**: Divide data into training and validation sets
4. **Model structure definition**: Define the grey-box model structure
5. **Optimization setup**: Define objectives and solvers
6. **Model training**: Estimate model parameters
7. **Model evaluation**: Validate the model against test data
8. **Results analysis**: Analyze model performance and predictions

Let's go through each step in detail.

## Step 1: Data Acquisition and Preprocessing

We'll start by downloading example data and loading it into a pandas DataFrame:

```python
import requests
import pandas as pd
import buildingsysid as bid

# Download example data from GitHub
def download_example_data(url, filename):
    """Download example data from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        with open(filename, "wb") as f:
            f.write(response.content)
        
        print(f"Example data downloaded successfully to {filename}!")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

# URL for example data
data_url = "https://raw.githubusercontent.com/michael666nan/buildingsysid/master/tutorial_data/Eplus_Project_Room_prbs.csv"
data_filename = "Eplus_Project_Room_prbs.csv"

# Download the data
download_example_data(data_url, data_filename)

# Read the CSV file with time stamps
df = pd.read_csv(data_filename, index_col='time_stamp', parse_dates=True) 

# Extract a specific time period for the analysis
# For example, selecting data from January
start_date = '2023-01-01'
end_date = '2023-01-31'
df = df.loc[start_date:end_date]

# Get the timestamps for later use
timestamps = df.index
```

This dataset contains:
- Indoor temperature measurements
- Outdoor temperature
- Solar gain
- Heating power

We filter the data to include only January 2023 for this example.

## Step 2: Creating IDData Objects for System Identification

Next, we prepare the data in the format required by the `buildingsysid` package:

```python
# Define output variable (what we want to predict)
output_name = ["Indoor Temp"]                
output_unit = ["°C"]                     
output_data = df[output_name].to_numpy().T                  

# Define input variables (what drives the system)
input_names = ["Outdoor Temp", "Solar Gain", "Heat Power"]  
input_units = ["°C", "W/m²", "W"]                        
input_data = df[input_names].to_numpy().T                   

# Create IDData with original sampling time
original_sampling_time = 60   # 60 seconds (1 minute)
data = bid.IDData(
       y=output_data,                    # Output data (indoor temperature)
       u=input_data,                     # Input data (outdoor temp, solar, heating)
       y_names=output_name,              # Name of output variable
       u_names=input_names,              # Names of input variables
       y_units=output_unit,              # Unit of output variable
       u_units=input_units,              # Units of input variables
       samplingTime=original_sampling_time,  # Sampling time in seconds
       timestamps=timestamps             # Time stamps for data points
)  

# Resample data to reduce computational complexity
model_sampling_time = 3600  # seconds (1 hour)
data_resampled = data.resample(
    new_sampling_time=model_sampling_time,
    output_agg="first",   # Take first output value in each interval
    input_agg="mean"      # Take mean of input values in each interval
)
```

We create an `IDData` object with our input and output variables. The original data has a sampling time of 1 minute, but we resample it to 1 hour to reduce computational complexity.

## Step 3: Splitting Data

We split the data into training and validation sets:

```python
# Split data - 70% for training, 30% for validation
train, val = data_resampled.split(train_ratio=0.7)

# Optional: Visualize the data
train.plot_timeseries(title="Training Data")
val.plot_timeseries(title="Validation Data")
train.plot_cross_correlation()
```

The training data is used to estimate model parameters, while the validation data is used to evaluate model performance.

## Step 4: Defining Grey-Box Model Structures

Now we define the model structure. We'll start with an unconstrained first-order grey-box model, then improve it with parameter constraints to address identifiability issues.

```python
# First, let's try a basic first-order model without constraints
grey1_unconstrained = bid.model_set.grey.First()

# Inspect the parameter dictionary using pretty printing for better readability
print("Unconstrained model parameter dictionary:")
import pprint
pp = pprint.PrettyPrinter(indent=4, width=100)
pp.pprint(grey1_unconstrained.param_dict)
```

### Improving Model Identifiability

When working with grey-box models, it's common to encounter identifiability issues where parameters have very large confidence intervals (sometimes including zero). This indicates that the model has too many degrees of freedom relative to the information in the data.

We can address this by fixing certain parameters based on physical knowledge:

```python
# Define fixed parameters to improve identifiability
# By fixing wh1 to 1.0, we assume all heat is stored in the room air capacity
fixed_params = {
    "wh1": 1.0  # Fix heat input coefficient
}

# Create the constrained model
grey1 = bid.model_set.grey.First(fixed_params=fixed_params)

# Inspect the constrained parameter dictionary
print("\nConstrained model parameter dictionary (with fixed wh1):")
pp.pprint(grey1.param_dict)
```

The first-order model represents a simple thermal model with one state variable (the building temperature). It captures:
- Heat exchange between indoor and outdoor environments
- Effect of solar radiation on indoor temperature
- Heat input from the heating system

By fixing `wh1` to 1.0, we're essentially normalizing the model with respect to the thermal capacity, which improves parameter identifiability.

## Step 5: Setting Up Optimization

Next, we define the objective function and solver for parameter estimation:

```python
# Define one-step-ahead prediction as our objective
onestep_pred = bid.criterion_of_fit.StandardObjective(kstep=1)

# Configure the least squares solver
ls_solver = bid.calculate.LeastSquaresSolver(
    method='trf',    # Trust Region Reflective algorithm
    verbose=2,       # Show detailed output during optimization
    ftol=1e-6        # Function tolerance for convergence
)

# Configure the first-order model optimization
opt_problem = bid.calculate.OptimizationManager(
    model_structure=grey1,     # The first-order model structure
    data=train,                # Training data
    objective=onestep_pred,    # Prediction objective
    solver=ls_solver           # Optimization solver
) 
```

We use one-step-ahead prediction as our objective function, which means the model will be optimized to predict the next temperature value given the current state and inputs. We use the Trust Region Reflective algorithm for optimization.

## Step 6: Training the Model

Now we estimate the model parameters:

```python
# Train the first-order model
grey1_opt = opt_problem.solve(
    initialization_strategies=["black_box"]  # Start with values typical for building systems
)

# Display estimated parameters and confidence intervals
print("\nFirst-order model parameters and confidence intervals:")
grey1_opt.print()

# Convert optimized model to state-space form
ss1 = grey1_opt.get_state_space()
```

The `solve` method estimates the model parameters that best fit the training data. We use the "black_box" initialization strategy, which starts with parameter values that are typical for building systems.

## Step 7: Evaluating Model Performance

We evaluate the model performance using the validation data:

```python
# Create a list with models for comparison
model_list = [ss1]
model_names = ["First-order"]

# Evaluate one-step ahead prediction (short-term forecast)
fit_1step, y_sim_1step = bid.validation.compare(
    model_list,                 # The models to compare
    val,                        # Validation data
    kstep=1,                    # One-step ahead prediction
    model_names=model_names,    # For plotting legend
    title="One-step Predictions"
)

# Evaluate 12-step ahead prediction (medium-term forecast, 12 hours)
fit_12step, y_sim_12step = bid.validation.compare(
    model_list,                 # The models
    val,                        # Validation data
    kstep=12,                   # 12-step ahead prediction
    model_names=model_names,    
    title="12-step Predictions (12 hours ahead)"
)

# Evaluate full simulation (long-term behavior)
fit_sim, y_sim_full = bid.validation.compare(
    model_list,                 # The models
    val,                        # Validation data
    model_names=model_names,
    title="Full Simulation"
)
```

We evaluate the model performance in three ways:
1. **One-step ahead prediction**: Short-term forecast (1 hour ahead)
2. **12-step ahead prediction**: Medium-term forecast (12 hours ahead)
3. **Full simulation**: Long-term behavior prediction

## Step 8: Results and Analysis

Finally, we analyze the results:

```python
# Display fit metrics for each model and prediction horizon
print("\nModel performance summary (fit %):")
print(f"{'Model':<15} {'1-step':<10} {'12-step':<10} {'Simulation':<10}")
print("-" * 45)
for i, name in enumerate(model_names):
    print(f"{name:<15} {fit_1step[i]:<10.2f} {fit_12step[i]:<10.2f} {fit_sim[i]:<10.2f}")

# Optional: Display more detailed model information as a dictionary
print("\nDetailed model parameters:")
import json
print(json.dumps(grey1_opt.get_param_dict(), indent=4, sort_keys=True))

# Create a formatted table of parameter values and confidence intervals
try:
    params = grey1_opt.get_param_dict()
    conf_intervals = grey1_opt.get_confidence_intervals()
    
    print("\nParameter values and confidence intervals:")
    print(f"{'Parameter':<15} {'Value':<10} {'Lower CI':<10} {'Upper CI':<10}")
    print("-" * 50)
    
    for param_name, value in params.items():
        if param_name in conf_intervals:
            lower_ci, upper_ci = conf_intervals[param_name]
            print(f"{param_name:<15} {value:<10.4f} {lower_ci:<10.4f} {upper_ci:<10.4f}")
        else:
            print(f"{param_name:<15} {value:<10.4f} {'N/A':<10} {'N/A':<10}")
except Exception as e:
    print(f"Could not create confidence interval table: {e}")
```

### Interpreting the Results

The fit percentages indicate how well the model predictions match the actual data. A higher percentage indicates better performance.

When examining the confidence intervals:
- Narrower intervals indicate more certain parameter estimates
- By fixing `wh1` to 1.0, we should see narrower confidence intervals for the remaining parameters compared to an unconstrained model
- The parameter `Ci` represents the thermal capacity of the building zone
- The parameter `Ri` represents the thermal resistance between indoor and outdoor

If confidence intervals are very wide or include zero, this might indicate model identifiability issues. Our constrained model with fixed `wh1` should help address this by reducing the degrees of freedom in the model.

You can also save the model for later use with the `save` method.

## Extending the Tutorial

This tutorial demonstrates a basic workflow with a constrained first-order model. Here are some ways to extend it:

### Comparing Constrained vs. Unconstrained Models

You can compare the performance and parameter identifiability between constrained and unconstrained models:

```python
# Create both unconstrained and constrained models
grey1_unconstrained = bid.model_set.grey.First()
grey1_constrained = bid.model_set.grey.First(fixed_params={"wh1": 1.0})

# Train both models
# (follow the same steps for both models)

# Convert to state-space
ss1_unconstrained = grey1_unconstrained_opt.get_state_space()
ss1_constrained = grey1_constrained_opt.get_state_space()

# Compare performance
model_list = [ss1_unconstrained, ss1_constrained]
model_names = ["Unconstrained", "Constrained"]

# Run comparison with validation data
# (as shown in the main tutorial)

# Compare confidence intervals between the two models to see the
# improvement in parameter identifiability
```

### Using Higher-Order Models

You can try a second-order model, which has two thermal masses:

```python
# Create a second-order grey-box model with constraints
fixed_params_2nd = {
    "wh1": 1.0,  # Fix heat input coefficient for first mass
    "wh2": 0.0   # Assume no direct heat input to second mass
}

grey2 = bid.model_set.grey.Second(fixed_params=fixed_params_2nd)

# Configure and train the second-order model
# (similar steps as for the first-order model)
```

### Exploring Parameter Bounds

You can add bounds to parameters based on physical knowledge:

```python
# Define fixed parameters
fixed_params = {
    "wh1": 1.0   # Fixed heat transfer coefficient
}

# Define parameter bounds
param_bounds = {
    "Ci": [1e4, 1e7],   # Thermal capacity bounds (J/K)
    "Ri": [1e-3, 1.0],  # Thermal resistance bounds (K/W)
    "x1[0]": [15, 25]   # Initial temperature between 15 and 25°C
}

# Create model with constraints and bounds
grey1_bounded = bid.model_set.grey.First(
    fixed_params=fixed_params,
    param_bounds=param_bounds
)
```

### Physical Interpretation of Parameters

An important extension is to interpret the physical meaning of the estimated parameters:

```python
# Get parameters
params = grey1_opt.get_param_dict()

# Calculate derived physical properties
thermal_capacity = params["Ci"]  # J/K
thermal_resistance = params["Ri"]  # K/W
time_constant = thermal_capacity * thermal_resistance  # seconds

# Convert to more intuitive units
time_constant_hours = time_constant / 3600  # hours
heat_loss_coefficient = 1 / thermal_resistance  # W/K

print(f"Building thermal capacity: {thermal_capacity:.2e} J/K")
print(f"Thermal resistance: {thermal_resistance:.4f} K/W")
print(f"Heat loss coefficient: {heat_loss_coefficient:.2f} W/K")
print(f"Building time constant: {time_constant_hours:.2f} hours")
```

## References

- [Building System Identification documentation](https://github.com/michael666nan/buildingsysid)
- Bacher, P., & Madsen, H. (2011). Identifying suitable models for the heat dynamics of buildings. Energy and Buildings, 43(7), 1511-1522.
- Reynders, G., Diriken, J., & Saelens, D. (2014). Quality of grey-box models and identified parameters as function of the accuracy of input and observation signals. Energy and Buildings, 82, 263-274.

---

## Conclusion

This tutorial has demonstrated how to use the `buildingsysid` package for grey-box system identification of building thermal dynamics. The resulting model can be used for temperature prediction, energy optimization, and HVAC control.

The first-order model provides a simple yet effective representation of building thermal dynamics. For more complex buildings, higher-order models might be more appropriate.

By following this workflow, you can create accurate thermal models for your own building data.