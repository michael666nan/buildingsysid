# Tutorial 1: Black-Box System Identification for Building Thermal Modeling

## Table of Contents
1. [Introduction to System Identification](#introduction-to-system-identification)
2. [Black-Box Models vs. White-Box Models](#black-box-models-vs-white-box-models)
3. [Linear Time-Invariant (LTI) Systems](#linear-time-invariant-lti-systems)
4. [State-Space Models](#state-space-models)
5. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
6. [Prediction Errors and Model Validation](#prediction-errors-and-model-validation)
7. [Practical Considerations](#practical-considerations)
8. [Advanced Topics](#advanced-topics)
9. [Exercises](#exercises)

## Introduction to System Identification

System identification is the process of developing mathematical models of dynamic systems based on observed input and output data. This tutorial demonstrates how to apply these techniques to building thermal modeling.

### Key Concepts:

- **System**: In this example, an EnergyPlus model of a single room is the underlying unknown process that generates the data 
- **Model**: A linear time-invariant state-space model is the mathematical representation we use to describe the system's behavior
- **Identification**: The process of determining model parameters from data
- **Validation**: Testing if the model accurately predicts system behavior

## Black-Box Models vs. White-Box and Grey-Box Models

### White-Box Models:
- Based on physical laws and first principles
- Parameters have physical meaning (e.g., thermal resistance, heat capacity)
- Require detailed knowledge of the system
- Example: RC-equivalent thermal networks for buildings

### Grey-Box Models:
- Hybrid approach with some physical structure but parameters identified from data
- Combine theoretical knowledge with empirical fitting
- Balance between physical interpretability and data-driven accuracy

### Black-Box Models:
- Purely data-driven without assuming physical structure
- Parameters lack direct physical interpretation
- Require minimal prior knowledge about the system
- Examples: ARX, ARMAX, state-space models, neural networks

In our example, we use black-box models because they:
1. Are versatile and applicable to many building types
2. Require less detailed building information
3. Can capture complex dynamics that may be missed by simplified physical models
4. Are computationally efficient for prediction and control

## Linear Time-Invariant (LTI) Systems

The models used in our example are Linear Time-Invariant (LTI) systems, which means:

- **Linear**: The output is proportional to the input (superposition principle applies)
- **Time-Invariant**: The system's response doesn't change over time

For building thermal modeling, LTI systems are often sufficient for:
- Fundamentally linear and time-invariant systems, or for small temperature ranges where nonlinear effects are minimal
- Building operation under normal conditions
- Short to medium-term predictions

### Mathematical Representation:

For a discrete-time LTI system in innovations form (state-space model with noise):

```
x(k+1) = A*x(k) + B*u(k) + K*e(k)
y(k) = C*x(k) + D*u(k) + e(k)
```

Where:
- x: State variable (internal temperature or energy)
- u: Inputs (outdoor temperature, solar gain, heating power)
- y: Outputs (indoor temperature)
- e: Innovations/prediction errors
- k: Time step
- A, B, C, D: System matrices to be identified
- K: Kalman gain matrix (also to be identified)

The innovations form is particularly important as it incorporates both the deterministic system dynamics (through A, B, C, D) and the stochastic components (through K and e). The Kalman gain K determines how much we should update our state estimates based on observed prediction errors.

## State-Space Models

State-space models are particularly useful for building thermal modeling because:

1. They can represent systems with multiple inputs and outputs
2. They capture the dynamics through internal state variables
3. They allow for different prediction horizons (one-step to simulation)
4. They're compatible with modern control techniques

### Model Order:

The "order" of a state-space model refers to the number of state variables:

- **First-order**: One state variable, typically representing the average building temperature
- **Second-order**: Two state variables, potentially representing different thermal zones or fast/slow dynamics
- **Higher-order**: More state variables for more complex systems

The choice of model order involves a tradeoff between:
- Simplicity and computational efficiency (lower order)
- Accuracy and ability to capture complex dynamics (higher order)

In our example, we compare first-order and second-order models to evaluate this tradeoff.

## Step-by-Step Code Walkthrough

Let's examine each section of the provided example code:

### Step 0: Installation (Before Running the Script)

Before using the `buildingsysid` library, you need to install it:

```bash
# Install the library
pip install buildingsysid
```

**Explanation:**
- This step is performed once in your environment, not as part of the script
- You may want to use a virtual environment to manage dependencies
- Check the library documentation for specific version requirements

### Step 1: Accessing Example Data

You'll need to download the example data file used in this tutorial:

```
Eplus_Project_Room_prbs.csv - Building simulation data from EnergyPlus
```

**Where to find the data:**
- The example data file is available in the GitHub repository: [github.com/michael666nan/buildingsysid/tree/master/tutorial_data/](https://github.com/michael666nan/buildingsysid/tree/master/tutorial_data/)
- Download this file to your working directory before running the script
- Alternatively, you can use the following command to download it directly from GitHub:

```python
# Optional: Download example data directly
import requests

# Use the RAW GitHub URL for direct file access
url = "https://raw.githubusercontent.com/michael666nan/buildingsysid/master/tutorial_data/Eplus_Project_Room_prbs.csv"

# Download and save the file
try:
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors
    
    with open("Eplus_Project_Room_prbs.csv", "wb") as f:
        f.write(response.content)
    
    print("Example data downloaded successfully!")
except Exception as e:
    print(f"Error downloading file: {e}")
```

### Step 2: Importing Required Libraries

First, import the necessary libraries for the analysis:

```python
# Import the necessary libraries
import buildingsysid as bid
import pandas as pd
import matplotlib.pyplot as plt  # For visualization
import numpy as np  # For numerical operations
```

**Explanation:**
- The `buildingsysid` library provides tools for system identification in building applications
- `pandas` is used for data handling and manipulation
- `matplotlib` for visualizing results and data inspection
- `numpy` for numerical calculations when needed

### Step 3: Loading and Preparing Data

```python
# Read the CSV file with time stamps
df = pd.read_csv('Eplus_Project_Room_prbs.csv', index_col='time_stamp', parse_dates=True) 

# Extract a specific time period for the analysis
# For example, selecting data from January
start_date = '2023-01-01'
end_date = '2023-01-31'
df = df.loc[start_date:end_date]

# Get the timestamps for later use
timestamps = df.index

# Define output variable (what we want to predict)
output_name = ["Indoor Temp"]                
output_unit = ["°C"]                     
output_data = df[output_name].to_numpy().T                  

# Define input variables (what drives the system)
input_names = ["Outdoor Temp", "Solar Gain", "Heat Power"]  
input_units = ["°C", "W/m²", "W"]                        
input_data = df[input_names].to_numpy().T                   
```

**Explanation:**
- The data comes from an EnergyPlus simulation, which is a common building energy simulation software
- We define what we want to predict (output) - indoor temperature
- We define what drives the system (inputs) - outdoor temperature, solar gain, and heating power
- **Time period selection**: We extract a specific time window from the dataset to focus our analysis
  - This is useful for analyzing seasonal effects or focusing on specific periods of interest
  - You can adjust the start_date and end_date to select different periods
- The `.T` operation transposes the arrays to match the expected format for the library
- Setting `parse_dates=True` ensures timestamps are properly converted to datetime objects
- The data is organized in time series with inputs that affect the building temperature

### Step 4: Creating the Identification Data Object

```python
# Create IDData
sampling_time = 60   # 60 seconds (1 minute)
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
```

**Explanation:**
- `IDData` is a specialized data container for system identification
- It keeps track of inputs, outputs, sampling time, and timestamps
- This structured format makes subsequent analysis more efficient
- The sampling time (60 seconds) represents the time between consecutive measurements

### Step 5: Resampling Data

```python
# Resample data
model_sampling_time = 3600  # seconds (1 hour)
data_resampled = data.resample(
    new_sampling_time=model_sampling_time,
    output_agg="first",   # Take first output value in each interval
    input_agg="mean"      # Take mean of input values in each interval
)
```

**Explanation:**
- Resampling from 1-minute to 1-hour intervals serves several purposes:
  1. Reduces noise in the data
  2. Decreases computational complexity
  3. Focuses on slow thermal dynamics (buildings typically respond slowly)
  4. Makes the model more suitable for hourly control decisions
- Different aggregation methods are used:
  - "first" for outputs means we take the temperature at the beginning of each hour
  - "mean" for inputs averages the driving forces over the hour

### Step 6: Splitting Data

```python
# Split data - 70% for training, 30% for validation
train, val = data_resampled.split(train_ratio=0.7)
```

**Explanation:**
- We use 70% of data for training and 30% for validation
- This is a common split ratio for system identification
- Training data is used to estimate model parameters
- Validation data is kept separate to evaluate model performance

### Step 7: Data Inspection and Visualization

```python
# Visualize the training data
train.plot_timeseries(title="Training Data")

# Visualize the validation data
val.plot_timeseries(title="Validation Data")

# Analyze correlations in the data
train.plot_cross_correlation()
```

**Explanation:**
- Plotting the time series helps visualize the data and check for anomalies
- Inspecting both training and validation sets ensures they capture similar dynamics
- Cross-correlation plots show how inputs affect outputs over time
  - They help identify which inputs have the strongest influence
  - They reveal time delays between inputs and outputs

### Step 8: Defining Model Structure

```python
# Define models structures
black1 = bid.model_set.black.First()   # first-order model
black2 = bid.model_set.black.Second()   # second-order model
```

**Explanation:**
- We create two model structures with different complexity:
  - First-order: Has one internal state variable
  - Second-order: Has two internal state variables
- These are "black-box" models because we don't impose physical structure
- The library handles the mathematical formulation internally

### Step 9: Defining Prediction Objective

```python
# Define one-step-ahead prediction as our objective
onestep_pred = bid.criterion_of_fit.StandardObjective(kstep=1)
```

**Explanation:**
- `StandardObjective` defines what error we want to minimize
- `kstep=1` specifies one-step-ahead prediction as our objective
- This means we want to predict the next time step given current data
- One-step prediction is often used for parameter estimation because it's mathematically convenient
- Other values could be used for multi-step (e.g. kstep=12)
- Skip the kstep argument for simulation objective

### Step 10: Configuring the Optimization Solver

```python
# Configure the least squares solver
ls_solver = bid.calculate.LeastSquaresSolver(
    method='trf',    # Trust Region Reflective algorithm
    verbose=2,       # Show detailed output during optimization
    ftol=1e-6        # Function tolerance for convergence
)
```

**Explanation:**
- The least squares solver finds parameters that minimize prediction errors
- `method='trf'` uses the Trust Region Reflective algorithm, which is efficient for constrained problems
- `verbose=2` provides detailed output to monitor the optimization progress
- `ftol=1e-6` sets the function tolerance for convergence (when to stop the optimization)

### Step 11: Setting Up the Optimization Problem

```python
# Configure the first-order model optimization
opt_problem1 = bid.calculate.OptimizationManager(
    model_structure=black1,    # The first-order model structure
    data=train,                # Training data
    objective=onestep_pred,    # Prediction objective
    solver=ls_solver           # Optimization solver
)    

# Configure the second-order model optimization (similar process)
opt_problem2 = bid.calculate.OptimizationManager(
    model_structure=black2,    # The second-order model structure 
    data=train,                # Training data
    objective=onestep_pred,    # Same prediction objective
    solver=ls_solver           # Same optimization solver
)
```

**Explanation:**
- `OptimizationManager` combines model, data, objective, and solver into a complete problem
- We create separate optimization problems for each model structure
- Both problems use the same training data, objective function, and solver
- This setup allows us to compare different model orders with consistent methodology

### Step 12: Solving the Optimization Problem (Training the Models)

```python
# Train the first-order model
black1_opt = opt_problem1.solve(
    initialization_strategies=["black_box"]  # Start with values typical for building systems
)

# Train the second-order model
black2_opt = opt_problem2.solve(
    initialization_strategies=["black_box"]  # Same initialization strategy
)
```

**Explanation:**
- The `solve()` method trains the model by finding optimal parameters
- `initialization_strategies=["black_box"]` uses starting values suitable for building thermal systems
- The solver iteratively adjusts parameters to minimize the objective function
- This process is repeated for both model structures
- The optimization results contain the best-fit parameters and additional information

### Step 13: Analyzing Model Parameters

```python
# Display estimated parameters and confidence intervals for both models
print("First-order model parameters:")
black1_opt.print()

print("Second-order model parameters:")
black2_opt.print()

# Convert optimized models to state-space form
ss1 = black1_opt.get_state_space()
ss2 = black2_opt.get_state_space()

# Create a list with models for comparison
model_list = [ss1, ss2]
model_names = ["First-order", "Second-order"]
```

**Explanation:**
- `print()` shows the estimated parameters with confidence intervals
- Narrow confidence intervals indicate more reliable parameter estimates
- `get_state_space()` converts the identified model to state-space form for simulation
- Creating a list of models allows for easy comparison in the validation step

### Step 14: Validating Model Performance

```python
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
    model_list,                     # The models
    val,                            # Validation data
    model_names=model_names,
    title="Full Simulation"
)

# Display fit metrics for each model and prediction horizon
print("\nModel performance (fit %):")
print(f"{'Model':<15} {'1-step':<10} {'12-step':<10} {'Simulation':<10}")
print("-" * 45)
for i, name in enumerate(model_names):
    print(f"{name:<15} {fit_1step[i]:<10.2f} {fit_12step[i]:<10.2f} {fit_sim[i]:<10.2f}")
```

**Explanation:**
- Validation compares predictions against actual data on the unseen validation set
- Multiple prediction horizons are tested to evaluate different aspects:
  - One-step (1 hour ahead): Tests immediate prediction accuracy
  - 12-step (12 hours ahead): Tests medium-term forecasting ability
  - Simulation (infinite-step): Tests long-term behavior stability
- The fit metric is typically a normalized RMSE, with 100% indicating perfect fit
- Comparing different model orders shows the tradeoff between complexity and accuracy
- Typically, fit decreases as prediction horizon increases

### Step 15: Saving Models for Future Use

```python
# Save models to files for later use
ss1.save("first_order")
ss2.save("second_order")

print("Models saved successfully!")
```

**Explanation:**
- The `save()` method stores the identified model to disk
- Saved models can be loaded later for predictions or control applications
- This allows you to identify models once and reuse them many times
- Models are saved with their complete structure and parameters
- You can load these models later using `bid.load_model("model_name")`

## Prediction Errors and Model Validation

### Types of Prediction Errors

When identifying system models, we can minimize different types of errors, each with distinct properties:

#### 1. One-Step Prediction Error
- **Definition**: Error when predicting one time step ahead using all past measured outputs
- **Mathematical representation**: e(k) = y(k) - ŷ(k|k-1)
  - Where ŷ(k|k-1) is the prediction of y(k) using data up to time k-1
- **Properties**:
  - Easiest error to minimize
  - Reflects the model's ability to capture immediate dynamics
  - Often used in parameter estimation algorithms
  - Usually results in the smallest error magnitude
  - May hide issues with long-term prediction capability

#### 2. Multi-Step Prediction Error
- **Definition**: Error when predicting multiple steps ahead using past measurements
- **Mathematical representation**: e_m(k) = y(k) - ŷ(k|k-m)
  - Where ŷ(k|k-m) is the prediction of y(k) using data up to time k-m
- **Properties**:
  - More challenging to minimize than one-step
  - Better represents medium-term forecasting ability
  - Errors tend to accumulate with prediction horizon
  - Good compromise between one-step and simulation error
  - Particularly useful for Model Predictive Control applications

#### 3. Simulation Error (Infinite-Step Prediction)
- **Definition**: Error when simulating the entire output sequence using only initial conditions and inputs
- **Mathematical representation**: e_sim(k) = y(k) - ŷ_sim(k)
  - Where ŷ_sim(k) is the simulated output using only input data and initial states
- **Properties**:
  - Most difficult error to minimize
  - Best represents long-term model behavior
  - Provides strongest test of model adequacy
  - Small simulation errors indicate a truly accurate model
  - Often larger in magnitude than other error types

The choice of which error to minimize depends on the intended use of the model:
- For state estimation or filtering: One-step prediction error
- For Model Predictive Control: Multi-step prediction error
- For long-term simulation: Simulation error

### Model Validation Techniques

Proper validation is crucial to ensure model reliability:

#### 1. One-Step Prediction Validation
- Predicts the next time step given current data
- Tests the model's ability to capture immediate dynamics
- Usually gives the best performance metrics

#### 2. Multi-Step Prediction Validation
- Predicts multiple steps into the future
- Tests medium-term forecasting ability
- More challenging as errors accumulate

#### 3. Full Simulation Validation
- Uses only initial conditions and inputs to predict all outputs
- Tests long-term behavior and stability
- Most rigorous validation method

#### 4. Residual Analysis
- Examines the prediction errors (residuals)
- Checks if residuals are random (white noise)
- Non-random patterns indicate model inadequacies
- Statistical tests: autocorrelation, cross-correlation with inputs

#### 5. Fitness Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Fit percentage (normalized RMSE)
- AIC/BIC (for model order selection)

## Practical Considerations

### Sampling Time Selection
- Match the dynamics of the system (buildings are slow, typically hourly is sufficient)
- Consider control application requirements
- Balance between data volume and information content

### Input Selection
- Include all significant drivers of the system
- Too few inputs: poor explanatory power
- Too many inputs: risk of overfitting

### Model Order Selection
- Start simple (first-order) and increase complexity if needed
- Use validation metrics to compare different orders
- Consider the Akaike Information Criterion (AIC) or similar metrics

### Seasonality and Weather Variation
- Ensure training data covers different conditions
- Consider separate models for different seasons if dynamics vary significantly
- Test robustness by validating across different weather patterns

## Advanced Topics

### Nonlinear Models
For systems with significant nonlinearities:
- Hammerstein-Wiener models
- Neural networks
- Nonlinear ARX models

### Time-Varying Parameters
For systems that change over time:
- Recursive identification
- Adaptive models
- Sliding window approaches

### Multiple Zones
For buildings with distinct thermal zones:
- MIMO (Multiple-Input Multiple-Output) models
- Coupled state-space representations
- Hierarchical modeling

### Model Predictive Control (MPC)
Using identified models for control:
- Prediction of future behavior
- Optimization of control actions
- Handling of constraints
- Trade-off between comfort and energy

## Exercises

1. **Model Order Comparison**: Modify the example to compare different model orders (1st, 2nd, 3rd, 4th) and determine the optimal complexity using AIC or BIC criteria. Create a plot showing how the fit metrics change with increasing model order.

2. **Sampling Time Analysis**: Try different sampling times (15 minutes, 30 minutes, 1 hour) and analyze:
   - Impact on model accuracy for different prediction horizons
   - Computational efficiency tradeoffs
   - Ability to capture different dynamics

3. **Error Minimization Comparison**: Compare models trained by minimizing different error types:
   - One-step prediction error
   - Multi-step prediction error (e.g., 6-step)
   - Simulation error
   - Multi-step prediction error summed over the prediction horizon
       In this case you need to specify the argument sum_hor=True
       
4. **Level of Excitation**: Use data with less excitation and feedback:
   - Use the example data file 'heavyweight_room_bounded_setpoint.csv' available in the GitHub repository: [github.com/michael666nan/buildingsysid/tree/master/tutorial_data/](https://github.com/michael666nan/buildingsysid/tree/master/tutorial_data/)
   - This data is generated by the same EnergyPlus model, but with a PI-controller and comfort bounded setpoints
   - Inspect the data to assess the data quality
   - Train models and assess performance