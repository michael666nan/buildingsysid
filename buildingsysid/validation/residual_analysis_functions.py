import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf, ccf


def residual_statistics(residuals):
    """
    Calculate basic statistical properties of residuals.
    
    Parameters:
    -----------
    residuals : array-like
        The residuals from your state space model.
        
    Returns:
    --------
    dict
        Dictionary containing statistical properties.
    """
    residuals = np.asarray(residuals).flatten()
    
    # Calculate basic statistics
    mean = np.mean(residuals)
    variance = np.var(residuals)
    std_dev = np.std(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    
    # Calculate confidence interval for mean (95%)
    n = len(residuals)
    se = std_dev / np.sqrt(n)
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se
    
    # Create summary dictionary
    results = {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mean_ci_95': (ci_lower, ci_upper),
        'n_samples': n
    }
    
    return results


def plot_residual_statistics(residuals, figsize=(12, 10)):
    """
    Create plots to visualize the statistical properties of residuals.
    
    Parameters:
    -----------
    residuals : array-like
        The residuals from your state space model.
    figsize : tuple, optional
        Figure size for the plots.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots.
    """
    residuals = np.asarray(residuals).flatten()
    
    # Create figure with subplots directly
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Q-Q Plot
    stats.probplot(residuals, plot=ax1)
    ax1.set_title('Q-Q Plot')
    
    # Histogram
    ax2.hist(residuals, bins=30, density=True, alpha=0.6, color='skyblue')
    
    # Add normal distribution curve
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax2.plot(x, p, 'k', linewidth=2)
    ax2.set_title('Histogram with Normal Distribution')
    
    # Residual time series (using dots instead of line)
    ax3.scatter(range(len(residuals)), residuals, s=3, alpha=0.6, color='blue')
    ax3.axhline(y=0, color='r', linestyle='-')
    ax3.axhline(y=2*np.std(residuals), color='r', linestyle='--')
    ax3.axhline(y=-2*np.std(residuals), color='r', linestyle='--')
    ax3.set_title('Residual Time Series')
    
    # Box plot
    ax4.boxplot(residuals)
    ax4.set_title('Residual Box Plot')
    
    # Adjust layout
    fig.tight_layout()
    
    # After tight_layout, adjust the top margin for the suptitle
    fig.subplots_adjust(top=0.9)
    
    return fig


def normality_test(residuals):
    """
    Perform normality tests on residuals.
    
    Parameters:
    -----------
    residuals : array-like
        The residuals from your state space model.
        
    Returns:
    --------
    dict
        Dictionary containing normality test results.
    """
    residuals = np.asarray(residuals).flatten()
    
    # Shapiro-Wilk test
    shapiro_test = stats.shapiro(residuals)
    
    # D'Agostino's K^2 test
    dagostino_test = stats.normaltest(residuals)
    
    # Jarque-Bera test
    jarque_bera_test = stats.jarque_bera(residuals)
    
    # Anderson-Darling test
    anderson_test = stats.anderson(residuals, dist='norm')
    
    results = {
        'shapiro_wilk': {
            'statistic': shapiro_test[0],
            'p_value': shapiro_test[1],
            'normal_at_5%': shapiro_test[1] > 0.05
        },
        'dagostino_k2': {
            'statistic': dagostino_test[0],
            'p_value': dagostino_test[1],
            'normal_at_5%': dagostino_test[1] > 0.05
        },
        'jarque_bera': {
            'statistic': jarque_bera_test[0],
            'p_value': jarque_bera_test[1],
            'normal_at_5%': jarque_bera_test[1] > 0.05
        },
        'anderson_darling': {
            'statistic': anderson_test.statistic,
            'critical_values': anderson_test.critical_values,
            'significance_levels': anderson_test.significance_level
        }
    }
    
    return results


def autocorrelation(residuals, max_lag=40, alpha=0.05, figsize=(12, 6)):
    """
    Compute and plot autocorrelation of residuals.
    
    Parameters:
    -----------
    residuals : array-like
        The residuals from your state space model.
    max_lag : int, optional
        Maximum lag to consider.
    alpha : float, optional
        Significance level for confidence intervals.
    figsize : tuple, optional
        Figure size for the plots.
        
    Returns:
    --------
    tuple
        Figure, autocorrelation values, and Ljung-Box test results.
    """
    residuals = np.asarray(residuals).flatten()
    
    # Calculate autocorrelation
    acf_values = acf(residuals, nlags=max_lag, alpha=alpha)
    #pacf_values = pacf(residuals, nlags=max_lag, alpha=alpha)
    
    # Perform Ljung-Box test
    lags = list(range(1, max_lag + 1))
    lb_test = acorr_ljungbox(residuals, lags=lags)
    
    # In newer versions of statsmodels, lb_test is a DataFrame
    # In older versions, it's a tuple of Series
    if isinstance(lb_test, pd.DataFrame):
        lb_statistics = lb_test['lb_stat'].values
        lb_pvalues = lb_test['lb_pvalue'].values
    else:
        # Handle tuple return format (older statsmodels)
        lb_statistics = lb_test[0].values
        lb_pvalues = lb_test[1].values
    
    # Create visualization with subplots, but adjust figure to leave room for suptitle
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.3)
    gs.update(top=0.85)  # Reserve space for the suptitle that will be added by the calling function
    
    # Create axes using the GridSpec
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    
    # ACF plot - use title='' to avoid default title 
    plot_acf(residuals, lags=max_lag, alpha=alpha, ax=ax0, title='')
    ax0.set_title('Autocorrelation Function (ACF)')
    
    # PACF plot - use title='' to avoid default title
    plot_pacf(residuals, lags=max_lag, alpha=alpha, ax=ax1, title='')
    ax1.set_title('Partial Autocorrelation Function (PACF)')
    
    # Prepare Ljung-Box test results
    lb_results = pd.DataFrame({
        'lag': lags,
        'lb_statistic': lb_statistics,
        'lb_pvalue': lb_pvalues,
        'significant': lb_pvalues < alpha
    })
    
    return fig, acf_values, lb_results


def cross_correlation(residuals, inputs, max_lag=40, alpha=0.05, figsize=(12, 8)):
    """
    Calculate and plot cross-correlation between residuals and inputs.
    
    Parameters:
    -----------
    residuals : array-like
        The residuals from your state space model.
    inputs : array-like or dict of array-like
        The input variables used in your model.
    max_lag : int, optional
        Maximum lag to consider.
    alpha : float, optional
        Significance level for confidence intervals.
    figsize : tuple, optional
        Figure size for the plots.
        
    Returns:
    --------
    tuple
        Figure and cross-correlation values.
    """
    from statsmodels.tsa.stattools import ccf
    residuals = np.asarray(residuals).flatten()
    
    # Handle inputs as dictionary or numpy array
    if isinstance(inputs, dict):
        input_names = list(inputs.keys())
        input_data = [np.asarray(inputs[name]).flatten() for name in input_names]
    elif isinstance(inputs, np.ndarray):
        if inputs.ndim == 1:
            input_names = ['Input 1']
            input_data = [inputs]
        elif inputs.ndim == 2:
            # For your specific case where rows are inputs and columns are timepoints
            input_names = [f'Input {i+1}' for i in range(inputs.shape[0])]
            input_data = [inputs[i, :] for i in range(inputs.shape[0])]
        else:
            raise ValueError("Array inputs should be 1D or 2D")
    else:
        try:
            inputs = np.asarray(inputs)
            if inputs.ndim == 1:
                input_names = ['Input 1']
                input_data = [inputs]
            elif inputs.ndim == 2:
                # For your specific case where rows are inputs and columns are timepoints
                input_names = [f'Input {i+1}' for i in range(inputs.shape[0])]
                input_data = [inputs[i, :] for i in range(inputs.shape[0])]
            else:
                raise ValueError("Array inputs should be 1D or 2D")
        except:
            raise ValueError("Inputs must be a dictionary, numpy array, or convertible to numpy array")
    
    n_inputs = len(input_data)
    
    # Create plots
    n_rows = int(np.ceil(n_inputs / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    
    if n_rows == 1 and n_inputs == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    
    # Dictionary to store the exact values that are plotted
    plotted_data = {}
    
    # Calculate significance bounds
    n = len(residuals)
    conf_int = stats.norm.ppf(1 - alpha/2) / np.sqrt(n)
    
    for i, name in enumerate(input_names):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Ensure matching lengths
        min_len = min(len(residuals), len(input_data[i]))
        res = residuals[:min_len]
        inp = input_data[i][:min_len]
        
        # Calculate cross-correlation using statsmodels' ccf
        values = ccf(inp, res, adjusted=False)
        
        # Define lags for plotting
        lags = np.arange(-max_lag, max_lag + 1)
        
        # Ensure lags and values match in length
        if len(lags) > len(values):
            lags = np.arange(-len(values)//2, len(values)//2 + 1)
        elif len(lags) < len(values):
            mid = len(values) // 2
            values = values[mid - max_lag : mid + max_lag + 1]
        
        # Store the exact values and lags that will be plotted
        plotted_data[name] = {
            'values': values.copy(),  # Make a copy to avoid reference issues
            'lags': lags.copy()
        }
        
        # Plot
        ax.stem(lags, values, linefmt='b-', markerfmt='bo', basefmt='r-')
        ax.axhline(y=conf_int, linestyle='--', color='gray')
        ax.axhline(y=-conf_int, linestyle='--', color='gray')
        ax.set_title(f'Cross-correlation: Residuals vs {name}')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_inputs, n_rows * 2):
        row, col = i // 2, i % 2
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Create a dictionary of just the cross-correlation values for the return value
    ccf_values = {name: plotted_data[name]['values'] for name in plotted_data}
    
    # Check for significant correlations using EXACTLY what was plotted
    print("\nSignificant Cross-correlations:")
    any_significant = False
    
    for name, data in plotted_data.items():
        values = data['values']
        lags = data['lags']
        
        # Find where the absolute correlation exceeds the threshold
        significant_indices = np.where(np.abs(values) > conf_int)[0]
        
        if len(significant_indices) > 0:
            any_significant = True
            # Find the maximum absolute correlation
            max_abs_idx = np.argmax(np.abs(values))
            max_value = values[max_abs_idx]
            max_lag = lags[max_abs_idx]
            
            print(f"✗ {name}: Significant at lag {max_lag} (value: {max_value:.4f})")
    
    if not any_significant:
        print("✓ No significant cross-correlation between residuals and inputs")
    
    return fig, ccf_values


def independence_test(residuals, max_lag=20):
    """
    Test for independence (whiteness) of residuals.
    
    Parameters:
    -----------
    residuals : array-like
        The residuals from your state space model.
    max_lag : int, optional
        Maximum lag to use in tests.
        
    Returns:
    --------
    dict
        Dictionary containing test results.
    """
    residuals = np.asarray(residuals).flatten()
    
    # Ljung-Box test
    lags = list(range(1, max_lag + 1))
    lb_test = acorr_ljungbox(residuals, lags=lags)
    
    # In newer versions of statsmodels, lb_test is a DataFrame
    # In older versions, it's a tuple of Series
    if isinstance(lb_test, pd.DataFrame):
        lb_statistics = lb_test['lb_stat'].values
        lb_pvalues = lb_test['lb_pvalue'].values
    else:
        # Handle tuple return format (older statsmodels)
        lb_statistics = lb_test[0].values
        lb_pvalues = lb_test[1].values
    
    # McLeod-Li test for ARCH effects
    squared_residuals = residuals**2
    ml_test = acorr_ljungbox(squared_residuals, lags=lags)
    
    # Handle McLeod-Li test results based on statsmodels version
    if isinstance(ml_test, pd.DataFrame):
        ml_statistics = ml_test['lb_stat'].values
        ml_pvalues = ml_test['lb_pvalue'].values
    else:
        ml_statistics = ml_test[0].values
        ml_pvalues = ml_test[1].values
    
    # --- Runs test (above/below median) ---
    # This is a safer implementation to avoid overflow and numerical issues
    try:
        median = np.median(residuals)
        above_median = residuals > median
        n_runs = len(np.diff(above_median).nonzero()[0]) + 1
        n1 = np.sum(above_median)
        n2 = len(residuals) - n1
        
        # Implement runs test in a safer way to avoid overflow
        if n1 <= 10 or n2 <= 10 or n1 > 1000 or n2 > 1000:
            # For very imbalanced data or very large datasets, 
            # use a direct normal approximation which is more stable
            if n1 > 0 and n2 > 0:
                # Simpler formula for expected runs that's less prone to overflow
                expected_runs = 1.0 + 2.0 * n1 * n2 / (n1 + n2)
                
                # More stable calculation of variance using log operations for large numbers
                if n1 + n2 > 30:  # For larger sample sizes, use normal approximation
                    std_dev_runs = np.sqrt(2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2) / 
                                          ((n1 + n2)**2 * (n1 + n2 - 1.0)))
                    
                    if np.isfinite(std_dev_runs) and std_dev_runs > 0:
                        z_stat = (n_runs - expected_runs) / std_dev_runs
                        p_value = 2.0 * stats.norm.sf(abs(z_stat))
                    else:
                        z_stat = np.nan
                        p_value = np.nan
                else:
                    # For smaller samples, don't compute z-stat
                    z_stat = np.nan
                    p_value = np.nan
            else:
                expected_runs = np.nan
                z_stat = np.nan
                p_value = np.nan
        else:
            # Use statsmodels or scipy implementation for normal-sized datasets
            # This is a simplified, safer approach
            expected_runs = 1.0 + 2.0 * n1 * n2 / (n1 + n2)
            
            # Compute variance with careful handling of potential numerical issues
            numerator = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)
            denominator = (n1 + n2)**2 * (n1 + n2 - 1.0)
            
            if denominator > 0 and numerator > 0:
                var_runs = numerator / denominator
                std_dev_runs = np.sqrt(var_runs)
                
                if std_dev_runs > 0:
                    z_stat = (n_runs - expected_runs) / std_dev_runs
                    p_value = 2.0 * stats.norm.sf(abs(z_stat))
                else:
                    z_stat = np.nan
                    p_value = np.nan
            else:
                z_stat = np.nan
                p_value = np.nan
    except Exception as e:
        print(f"Warning: Error in runs test calculation: {e}")
        n_runs = np.nan
        expected_runs = np.nan
        z_stat = np.nan
        p_value = np.nan
    
    # Compile results
    results = {
        'ljung_box_test': {
            'lags': lags,
            'statistics': lb_statistics,
            'p_values': lb_pvalues,
            'reject_independence_5%': lb_pvalues < 0.05
        },
        'mcleod_li_test': {
            'lags': lags,
            'statistics': ml_statistics,
            'p_values': ml_pvalues,
            'reject_independence_5%': ml_pvalues < 0.05
        },
        'runs_test': {
            'n_runs': n_runs,
            'expected_runs': expected_runs,
            'z_statistic': z_stat,
            'p_value': p_value,
            'reject_randomness_5%': False if np.isnan(p_value) else p_value < 0.05
        }
    }
    
    return results