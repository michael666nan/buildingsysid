# Tutorial 1: Black-Box System Identification for Building Thermal Modeling

## Table of Contents
1. [Introduction to System Identification](#introduction-to-system-identification)
2. [Black-Box Models vs. White-Box Models](#black-box-models-vs-white-box-models)
3. [Linear Time-Invariant (LTI) Systems](#linear-time-invariant-lti-systems)
4. [State-Space Models](#state-space-models)
5. [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
6. [Model Validation Techniques](#model-validation-techniques)
7. [Practical Considerations](#practical-considerations)
8. [Advanced Topics](#advanced-topics)
9. [Exercises](#exercises)

## Introduction to System Identification

System identification is the process of developing mathematical models of dynamic systems based on observed input and output data.

### Key Concepts:

- **System**: In this example an EnergyPlus model of a single room is the underlying unknown process that generates the data 
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
# Install the library if you haven't done so
pip install buildingsysid
```

**Explanation:**
- This step is performed once in your environment, not as part of the script
- You may want to use a virtual environment to manage dependencies
- Check the library documentation for specific version requirements

### Step 1: Accessing Example Data

You'll need to download the example data file used in this tutorial:

```
heavyweight_room_prbs.csv - Building simulation data from EnergyPlus
```

**Where to find the data:**
- The example data file is available in the GitHub repository: [github.com/michael666nan/buildingsysid/tree/master/tutorial_data/](https://github.com/michael666nan/buildingsysid/tree/master/examples/data/)
- Download this file to your working directory before running the script
- Alternatively, you can use the following command to download it directly from GitHub:

```python
# Optional: Download example data directly (uncomment and adjust URL as needed)
# import requests
# url = "https://raw.githubusercontent.com/michael666nan/buildingsysid/tree/master/tutorial_data/heavyweight_room_prbs.csv"
# with open("heavyweight_room_prbs.csv", "wb") as f:
#     f.write(requests.get(url).content)
```

### Step 2: Importing Required Libraries

First, import the necessary libraries for the analysis:

```python
# Import the necessary libraries
import buildingsysid as bid
import pandas as pd
```

**Explanation:**
- The `buildingsysid` library provides tools for system identification in building applications
- `pandas` is used for data handling and manipulation


### Step 2: Loading and Preparing Data

### Step 2: Loading and Preparing Data

```python
# Read the CSV file with time stamps
df = pd.read_csv('heavyweight_room_prbs.csv', index_col='time_stamp', parse_dates=True) 

# Extract a specific time period for the analysis
# For example, selecting data from January to March
start_date = '2019-01-01 00:00:00'
end_date = '2019-03-31 23:59:59'
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

### Step 2: Creating the Identification Data Object

```python
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

### Step 3: Resampling Data

```python
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

### Step 4: Splitting Data

```python
train, val = data_resampled.split(train_ratio=0.7)
train.plot_timeseries(title="Training Data")
val.plot_timeseries(title="Validation Data")
train.plot_cross_correlation()
train.plot_partial_cross_correlation()
```

**Explanation:**
- We use 70% of data for training and 30% for validation
- Plotting the time series helps visualize the data and check for anomalies
- Cross-correlation plots show how inputs affect outputs over time
- Partial cross-correlation shows direct relationships after removing indirect effects

### Step 5: Defining Model Structure

```python
black1 = bid.model_set.black.First()   # first-order model
black2 = bid.model_set.black.Second()   # second-order model
```

**Explanation:**
- We create two model structures with different complexity:
  - First-order: Has one internal state variable
  - Second-order: Has two internal state variables
- These are "black-box" models because we don't impose physical structure
- The library handles the mathematical formulation internally

### Step 6 & 7: Defining Prediction Objective and Solver

```python
onestep_pred = bid.criterion_of_fit.StandardObjective(kstep=1)
ls_solver = bid.calculate.LeastSquaresSolver(
    method='trf',
    verbose=2,
    ftol=1e-6
)
```

**Explanation:**
- `kstep=1` specifies one-step-ahead prediction as our objective
- The least squares solver finds parameters that minimize prediction errors
- `method='trf'` uses the Trust Region Reflective algorithm
- `ftol=1e-6` sets the function tolerance for convergence

### Step 8 & 9: Setting Up and Solving the Optimization Problem

```python
opt_problem1 = bid.calculate.OptimizationManager(
    model_structure=black1,
    data=train,
    objective=onestep_pred,
    solver=ls_solver
)    

black1_opt = opt_problem1.solve(
    initialization_strategies=["black_box"]
)
```

**Explanation:**
- `OptimizationManager` combines model, data, objective, and solver
- The `solve()` method trains the model by finding optimal parameters
- `initialization_strategies=["black_box"]` uses starting values suitable for building systems
- A similar process is repeated for the second-order model

### Step 10-13: Analyzing Results and Validation

```python
black1_opt.print()
ss1 = black1_opt.get_state_space()
ss2 = black2_opt.get_state_space()

# Validate models
fit, y_sim = bid.validation.compare(
    model_list,
    val,
    kstep=1,
    model_names=model_names,
    title="One-step Predictions"
)
```

**Explanation:**
- `print()` shows the estimated parameters with confidence intervals
- `get_state_space()` converts the identified model to state-space form
- Validation compares predictions against actual data
- Multiple prediction horizons are tested:
  - One-step (1 hour ahead)
  - 12-step (12 hours ahead)
  - Simulation (long-term behavior)

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

1. Modify the example to compare different model orders (1st, 2nd, 3rd) and determine the optimal complexity.

2. Experiment with different input variables. How does the model performance change if you remove solar gain or add other inputs?

3. Try different sampling times (30 minutes, 2 hours) and analyze the impact on model accuracy.

4. Implement cross-validation to ensure model robustness.

5. Compare the performance of black-box models with grey-box models (if available in the library).

6. Design a Model Predictive Control strategy using the identified model to optimize heating control for energy savings.