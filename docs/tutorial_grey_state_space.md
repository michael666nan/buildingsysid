# Grey-Box System Identification for Building Thermal Modeling

This tutorial demonstrates a workflow for building thermal modeling using grey-box system identification techniques with the `buildingsysid` package.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Workflow Overview](#workflow-overview)
- [Step 1: Data Acquisition and Preprocessing](#step-1-data-acquisition-and-preprocessing)
- [Step 2: Creating IDData Objects for System Identification](#step-2-creating-iddata-objects-for-system-identification)
- [Step 3: Splitting Data](#step-3-splitting-data)
- [Step 4: Defining Unconstrained Model Structure](#step-4-defining-unconstrained-model-structure)
- [Step 5: Setting Up Optimization](#step-5-setting-up-optimization)
- [Step 6: Training the Unconstrained Model](#step-6-training-the-unconstrained-model)
- [Step 7: Evaluating Unconstrained Model Performance](#step-7-evaluating-unconstrained-model-performance)
- [Step 8: Addressing Identifiability Issues](#step-8-addressing-identifiability-issues)
- [Step 9: Training and Evaluating the Constrained Model](#step-9-training-and-evaluating-the-constrained-model)
- [Exercises](#exercises)
- [References](#references)

## Introduction
Grey-box modeling combines physical knowledge with data-driven techniques. This approach is particularly useful when you  are interested in learning the values of physical parameters such as heat transfer coefficients, thermal capacities or solar aperture.

Grey-box modeling can lead to more robust and generalizable models - compared to black-box models - in case there isn't a lot of high quality training data.

In this tutorial, we'll use the `buildingsysid` package to create a thermal model of a building zone based on temperature and energy data. The resulting model can predict indoor temperature based on outdoor conditions and heating inputs.

## Prerequisites
To follow this tutorial, you'll need to install buildingsysid:

You can install these packages using pip:
```bash
pip install buildingsysid
```

## Workflow Overview

The system identification workflow consists of the following steps:

1. Data acquisition and preprocessing
2. Creating IDData objects for system identification
3. Defining grey-box model structures (unconstrained first)
4. Setting up optimization objectives and solvers
5. Training the models
6. Analyzing identifiability issues
7. Fixing identifiability with parameter constraints
8. Evaluating and comparing models

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

This dataset contains time series data for a building zone, including:
- Indoor temperature
- Outdoor temperature
- Solar gain 
- Heating power

We filter the data to include only January 2023 for this example.

## Step 2: Creating IDData Objects for System Identification

Next, we prepare the data in the format required by the `buildingsysid` package. The `IDData` object is a container for input-output data used in system identification:

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
# Uncomment these lines if you want to see the data visualizations
# train.plot_timeseries(title="Training Data")
# val.plot_timeseries(title="Validation Data")
# train.plot_cross_correlation()
```

The training data is used to estimate model parameters, while the validation data is used to evaluate model performance.

## Step 4: Defining Unconstrained Model Structure

We'll start with an unconstrained first-order grey-box model. A first-order model represents the building as a single thermal mass with one state variable (the building temperature):

```python
# Create a first-order grey-box model without constraints
grey1_unconstrained = bid.model_set.grey.First()
```

The first-order grey-box model corresponds to a simple RC (Resistance-Capacitance) thermal network. The key parameters of this model include:

- `Ci`: Thermal capacity of the building [J/K]
- `Ri`: Thermal resistance between indoor and outdoor [K/W]
- `wh1`: Heat input coefficient [-]
- `ws1`: Solar gain coefficient [-]
- `x1[0]`: Initial temperature [°C]

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
```

We use one-step-ahead prediction as our objective function, which means the model will be optimized to predict the next temperature value given the current state and inputs.

## Step 6: Training the Unconstrained Model

Now we set up and solve the optimization problem to estimate the model parameters:

```python
# Configure the first-order model optimization
print("\nTraining the unconstrained model...")
opt_problem = bid.calculate.OptimizationManager(
    model_structure=grey1_unconstrained,  # The unconstrained model
    data=train,                           # Training data
    objective=onestep_pred,               # Prediction objective
    solver=ls_solver                      # Optimization solver
) 

# Train the first-order model
grey1_opt = opt_problem.solve(
    initialization_strategies=["grey_box"]  # Start with values typical for building systems
)

# Display estimated parameters and confidence intervals
grey1_opt.print()

# Convert optimized model to state-space form
ss1_unconstrained = grey1_opt.get_state_space()
```

The `solve` method estimates the model parameters that best fit the training data. We use the "grey_box" initialization strategy, which starts with parameter values that are typical for building systems.

## Step 7: Evaluating Unconstrained Model Performance

Let's evaluate the unconstrained model performance using the validation data:

```python
# One-step ahead prediction
fit_1step, _ = bid.validation.compare(
    ss1_unconstrained, val, kstep=1, make_plot=False
)  

# 12-step ahead prediction
fit_12step, _ = bid.validation.compare(
    ss1_unconstrained, val, kstep=12, make_plot=False
)  

# Simulation (No Feedback)
fit_sim, _ = bid.validation.compare(
    ss1_unconstrained, val, make_plot=False
)  

# Display fit metrics for unconstrained model
print("\nUnconstrained model performance summary (fit %):")
print(f"1-step: {fit_1step:.2f} %")
print(f"12-step: {fit_12step:.2f} %")
print(f"Simulation: {fit_sim:.2f} %")
```

We evaluate the model in three ways:
1. **One-step ahead prediction**: Short-term forecast (1 hour ahead)
2. **12-step ahead prediction**: Medium-term forecast (12 hours ahead)
3. **Full simulation**: Long-term behavior prediction

### Analyzing Identifiability Issues

At this point, you should examine the confidence intervals of the parameters. Wide confidence intervals, especially those that include zero, indicate identifiability issues. These issues occur because:

1. The model has more parameters than can be uniquely determined from the data
2. Different parameters have similar effects on the model output
3. The parameters are correlated

The printed output from `grey1_opt.print()` will show these wide confidence intervals for the unconstrained model.

## Step 8: Addressing Identifiability Issues

To address the identifiability issues, we'll create a constrained model by fixing certain parameters and adding bounds:

```python
# Define fixed parameters to improve identifiability
# By fixing wh1 to 1.0, we assume all heat is stored in the room air capacity
fixed_params = {
    "wh1": 1.0  # Fix heat input coefficient
}

# Create the constrained model
grey1_constrained = bid.model_set.grey.First(
    fixed_params=fixed_params
)
```

By fixing `wh1` to 1.0, we're essentially normalizing the model with respect to the thermal capacity, which improves parameter identifiability.

## Step 9: Training and Evaluating the Constrained Model

Now let's train and evaluate the constrained model:

```python
# Configure the constrained optimization problem
opt_problem_constrained = bid.calculate.OptimizationManager(
    model_structure=grey1_constrained,  # The constrained model
    data=train,                         # Training data
    objective=onestep_pred,             # Prediction objective
    solver=ls_solver                    # Optimization solver
) 

# Train the model
grey1_constrained_opt = opt_problem_constrained.solve(
    initialization_strategies=["grey_box"]
)

# Display estimated parameters and confidence intervals
grey1_constrained_opt.print()

# Convert optimized model to state-space form
ss1_constrained = grey1_constrained_opt.get_state_space()

# Evaluate constrained model at different prediction horizons
fit_1step_c, _ = bid.validation.compare(
    ss1_constrained, val, kstep=1, make_plot=False
)  # One-step ahead

fit_12step_c, _ = bid.validation.compare(
    ss1_constrained, val, kstep=12, make_plot=False
)  # 12-step ahead

fit_sim_c, _ = bid.validation.compare(
    ss1_constrained, val, make_plot=False
)  # Simulation

# Display fit metrics for constrained model
print("\nConstrained model performance summary (fit %):")
print(f"1-step: {fit_1step_c:.2f} %")
print(f"12-step: {fit_12step_c:.2f} %")
print(f"Simulation: {fit_sim_c:.2f} %")
```

### Comparing the Results

When examining the confidence intervals:
- Notice that the unconstrained model has much wider confidence intervals, possibly including zero for some parameters
- The constrained model should have narrower confidence intervals, indicating better parameter identifiability
- By fixing `wh1` to 1.0, we've normalized the model and made it more identifiable

The fit percentages should be similar between the two models, but the constrained model parameters have more physical meaning and reliability.

## Exercises

1. **Compare Model Predictions:** Add code to visualize and compare the predictions of both models:

```python
# Create a list with both models for comparison
model_list = [ss1_unconstrained, ss1_constrained]
model_names = ["Unconstrained", "Constrained (wh1=1.0)"]

# Compare models with validation data
bid.validation.compare(
    model_list,                 # The models to compare
    val,                        # Validation data
    kstep=1,                    # One-step ahead prediction
    model_names=model_names,    # For plotting legend
    title="One-step Predictions Comparison"
)
```

2. **Experiment with Parameter Bounds:** Try different bounds for the parameters to see how they affect the model identifiability and performance:

```python
# Define different parameter bounds
param_bounds_experiment = {
    "Ci": [1e4, 1e7],   # Thermal capacity bounds (J/K)
    "Ri": [1e-3, 1.0],  # Thermal resistance bounds (K/W)
    "k1": [0, 1],       # Constraint k1 to be between 0 and 1
}

# Create a new constrained model with these bounds
grey1_experiment = bid.model_set.grey.First(
    fixed_params={"wh1": 1.0},
    param_bounds=param_bounds_experiment
)

# Train and evaluate this model
# (Follow steps similar to the constrained model above)
```

3. **Try a Higher-Order Model:** Implement a second-order grey-box model and compare its performance with the first-order models:

```python
# Create a second-order grey-box model with constraints
fixed_params_2nd = {
    "wh1": 1.0,  # Fix heat input coefficient for first mass
    "wh2": 0.0   # Assume no direct heat input to second mass
}

grey2 = bid.model_set.grey.Second(fixed_params=fixed_params_2nd)

# Train and evaluate this model
# (Follow steps similar to the constrained model above)
```

## Conclusion

In this tutorial, we've demonstrated the workflow for grey-box system identification of building thermal dynamics. We started with an unconstrained model and observed identifiability issues through wide confidence intervals. We then addressed these issues by fixing parameters and adding bounds, resulting in a more identifiable model with narrower confidence intervals.

The first-order model provides a simple yet effective representation of building thermal dynamics. By fixing the parameter `wh1` to 1.0, we normalize the model with respect to thermal capacity, which helps resolve identifiability issues while maintaining model performance.

Understanding and addressing identifiability issues is critical in building thermal modeling to obtain reliable and physically meaningful parameter estimates.
