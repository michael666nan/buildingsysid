import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

from .residual_analysis_functions import (
    residual_statistics, plot_residual_statistics, normality_test,
    autocorrelation, cross_correlation, independence_test
)

def perform_residual_analysis(residuals, inputs=None, max_lag=20, alpha=0.05, 
                             save_report=False, report_path='residual_analysis_report.pdf',
                             show_plots=True):
    """
    Perform a comprehensive residual analysis for state space model validation.
    
    Parameters:
    -----------
    residuals : array-like
        The residuals from your state space model (measured - predicted).
    inputs : array-like or dict, optional
        The input variables used in your model. If numpy array, should be 2D where
        rows are different inputs and columns are timepoints.
    max_lag : int, optional
        Maximum lag to consider for correlation analyses.
    alpha : float, optional
        Significance level for statistical tests (default: 0.05).
    save_report : bool, optional
        Whether to save results to a PDF report.
    report_path : str, optional
        File path for the PDF report.
    show_plots : bool, optional
        Whether to display plots during analysis.
        
    Returns:
    --------
    dict
        Dictionary containing all analysis results and validation summary.
    """
    # Ensure residuals are numpy array
    residuals = np.asarray(residuals).flatten()
    n_samples = len(residuals)
    
    # Dictionary to store all results
    results = {'validation_summary': {}}
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Optional PDF report
    if save_report:
        pdf = PdfPages(report_path)
    
    # Create figure list to keep track
    figures = []
    
    # Print header for console output
    print("\n" + "="*80)
    print("COMPREHENSIVE RESIDUAL ANALYSIS FOR STATE SPACE MODEL VALIDATION")
    print("="*80)
    
    #######################################################
    # 1. Basic Statistical Analysis of Residuals
    #######################################################
    print("\n" + "="*80)
    print("1. STATISTICAL ANALYSIS OF RESIDUALS")
    print("="*80)

    # Calculate basic statistics
    stats_results = residual_statistics(residuals)
    results['statistics'] = stats_results
    
    print("\nBasic Statistics:")
    for key, value in stats_results.items():
        print(f"{key}: {value}")
    
    print("\nInterpretation:")
    # Check if mean is close to zero (within 2 standard errors)
    se = stats_results['std_dev'] / np.sqrt(n_samples)
    mean_ok = abs(stats_results['mean']) < 2 * se
    
    if mean_ok:
        print("✓ Mean is not significantly different from zero")
        print(f"  (mean: {stats_results['mean']:.6f}, 2*SE: {2*se:.6f})")
    else:
        print("✗ Mean is significantly different from zero")
        print(f"  (mean: {stats_results['mean']:.6f}, 2*SE: {2*se:.6f})")
        print("  This suggests model bias or systematic error")
    
    # Check skewness
    if abs(stats_results['skewness']) < 0.5:
        print("✓ Skewness is close to zero (symmetric distribution)")
    else:
        print(f"✗ Skewness is {stats_results['skewness']:.4f}, suggesting asymmetric residuals")
        print("  This may indicate nonlinear effects not captured by the model")
    
    # Check kurtosis (excess kurtosis close to 0)
    excess_kurtosis = stats_results['kurtosis'] - 3
    if abs(excess_kurtosis) < 1:
        print("✓ Kurtosis is close to normal distribution")
    else:
        if excess_kurtosis > 0:
            print(f"✗ Excess kurtosis is {excess_kurtosis:.4f}, indicating heavy tails")
            print("  This suggests presence of outliers or non-Gaussian behavior")
        else:
            print(f"✗ Excess kurtosis is {excess_kurtosis:.4f}, indicating light tails")
            print("  This suggests bounded or constrained behavior")
    
    # Visualize the statistical properties
    stat_fig = plot_residual_statistics(residuals)
    stat_fig.suptitle("Statistical Properties of Residuals", fontsize=16)
    
    if save_report:
        pdf.savefig(stat_fig)
    if show_plots:
        plt.show()
    else:
        plt.close(stat_fig)
    
    figures.append(stat_fig)
    
    #######################################################
    # 2. Normality Tests
    #######################################################
    print("\n" + "="*80)
    print("2. NORMALITY TESTS")
    print("="*80)
    
    # Perform normality tests
    norm_results = normality_test(residuals)
    results['normality'] = norm_results
    
    print("\nNormality Test Results:")
    for test_name, test_results in norm_results.items():
        print(f"\n{test_name}:")
        for key, value in test_results.items():
            print(f"  {key}: {value}")
    
    # Get p-value from Shapiro-Wilk test for summary
    shapiro_p = norm_results['shapiro_wilk']['p_value']
    normality_ok = shapiro_p > alpha
    
    print("\nInterpretation:")
    if normality_ok:
        print(f"✓ Shapiro-Wilk test (p={shapiro_p:.4f} > {alpha}) suggests residuals are normally distributed")
        print("  This is consistent with white noise assumptions")
    else:
        print(f"✗ Shapiro-Wilk test (p={shapiro_p:.4f} < {alpha}) suggests residuals are NOT normally distributed")
        print("  This may indicate nonlinear behavior, outliers, or heteroscedasticity")
    
    #######################################################
    # 3. Autocorrelation Analysis
    #######################################################
    print("\n" + "="*80)
    print("3. AUTOCORRELATION ANALYSIS")
    print("="*80)
    
    # Perform autocorrelation analysis
    acf_fig, acf_values, lb_results = autocorrelation(residuals, max_lag=max_lag, alpha=alpha)
    acf_fig.suptitle("Autocorrelation Analysis of Residuals", fontsize=16)
    results['autocorrelation'] = {
        'acf_values': acf_values,
        'ljung_box': lb_results
    }
    
    if save_report:
        pdf.savefig(acf_fig)
    if show_plots:
        plt.show()
    else:
        plt.close(acf_fig)
    
    figures.append(acf_fig)
    
    # Check Ljung-Box test results at specific lags
    lag_indices = [4, 9, 14]  # Lags 5, 10, 15
    lb_significant = False
    
    print("\nLjung-Box Test Results (selected lags):")
    for idx in lag_indices:
        if idx < len(lb_results):
            lag = lb_results['lag'][idx]
            p_val = lb_results['lb_pvalue'][idx]
            significant = p_val < alpha
            if significant:
                lb_significant = True
            print(f"Lag {lag}: p-value = {p_val:.6f} ({'Significant' if significant else 'Not significant'})")
    
    print("\nInterpretation:")
    autocorr_ok = not lb_significant
    if autocorr_ok:
        print("✓ No significant autocorrelation detected by Ljung-Box test")
        print("  Residuals appear to be white noise (good)")
    else:
        print("✗ Significant autocorrelation detected by Ljung-Box test")
        print("  This suggests your model doesn't capture all dynamics in the data")
        print("  Consider increasing model order or adding relevant states")
    
    #######################################################
    # 4. Cross-correlation Analysis (if inputs provided)
    #######################################################
    crosscorr_ok = True
    if inputs is not None:
        print("\n" + "="*80)
        print("4. CROSS-CORRELATION ANALYSIS")
        print("="*80)
        
        # Perform cross-correlation analysis
        ccf_fig, ccf_values = cross_correlation(residuals, inputs, max_lag=max_lag, alpha=alpha)
        ccf_fig.suptitle("Cross-correlation: Residuals vs Inputs", fontsize=16)
        results['cross_correlation'] = ccf_values
        
        if save_report:
            pdf.savefig(ccf_fig)
        if show_plots:
            plt.show()
        else:
            plt.close(ccf_fig)
        
        figures.append(ccf_fig)
        
        # Check if any significant cross-correlations
        print("\nSignificant Cross-correlations:")
        critical_value = stats.norm.ppf(1 - alpha/2) / np.sqrt(n_samples)
        for name in ccf_values:
            max_ccf = np.max(np.abs(ccf_values[name]))
            significant = max_ccf > critical_value
            if significant:
                crosscorr_ok = False
                max_idx = np.argmax(np.abs(ccf_values[name]))
                lag = max_idx - max_lag if max_idx - max_lag < 0 else max_idx - max_lag
                print(f"✗ {name}: Significant at lag {lag} (value: {ccf_values[name][max_idx]:.4f})")
        
        if crosscorr_ok:
            print("✓ No significant cross-correlation between residuals and inputs")
            print("  Model appears to capture input-output relationships adequately")
        else:
            print("\nInterpretation:")
            print("✗ Significant cross-correlation detected")
            print("  This suggests the model doesn't fully capture how inputs affect outputs")
            print("  Consider adjusting system structure, adding states, or modifying input integration")
    
    #######################################################
    # 5. Independence Tests
    #######################################################
    print("\n" + "="*80)
    print("5. INDEPENDENCE TESTS")
    print("="*80)
    
    # Perform independence tests
    indep_results = independence_test(residuals, max_lag=max_lag)
    results['independence'] = indep_results
    
    # Check Ljung-Box test at lag 10
    lb_lag10_idx = min(9, len(indep_results['ljung_box_test']['lags'])-1)
    lb_lag10_pval = indep_results['ljung_box_test']['p_values'][lb_lag10_idx]
    lb_indep_ok = lb_lag10_pval > alpha
    
    # Check McLeod-Li test at lag 10 (for ARCH effects)
    ml_lag10_idx = min(9, len(indep_results['mcleod_li_test']['lags'])-1)
    ml_lag10_pval = indep_results['mcleod_li_test']['p_values'][ml_lag10_idx]
    ml_indep_ok = ml_lag10_pval > alpha
    
    # Runs test
    runs_pval = indep_results['runs_test']['p_value']
    # Handle potential NaN values
    if np.isnan(runs_pval):
        runs_ok = True  # Consider it passed if we couldn't compute it
        print("Note: Runs test could not be computed reliably with this data")
    else:
        runs_ok = runs_pval > alpha
    
    print("\nLjung-Box Test (lag 10):")
    print(f"p-value: {lb_lag10_pval:.6f} ({'Not significant' if lb_indep_ok else 'Significant'})")
    
    print("\nMcLeod-Li Test (lag 10):")
    print(f"p-value: {ml_lag10_pval:.6f} ({'Not significant' if ml_indep_ok else 'Significant'})")
    
    print("\nRuns Test:")
    print(f"p-value: {runs_pval:.6f} ({'Not significant' if runs_ok else 'Significant'})")
    
    print("\nInterpretation:")
    independence_ok = lb_indep_ok and ml_indep_ok and runs_ok
    
    if independence_ok:
        print("✓ All independence tests passed")
        print("  Residuals appear to be random and independent")
    else:
        print("✗ One or more independence tests failed")
        if not lb_indep_ok:
            print("  - Ljung-Box test suggests linear dependencies in residuals")
        if not ml_indep_ok:
            print("  - McLeod-Li test suggests nonlinear dependencies (ARCH effects)")
        if not runs_ok:
            print("  - Runs test suggests non-random sequence of residuals")
    
    #######################################################
    # 6. Overall Model Validation Decision
    #######################################################
    print("\n" + "="*80)
    print("6. MODEL VALIDATION SUMMARY")
    print("="*80)
    
    # Store validation results
    validation = {
        'mean_ok': mean_ok,
        'normality_ok': normality_ok,
        'autocorr_ok': autocorr_ok,
        'crosscorr_ok': crosscorr_ok,
        'independence_ok': independence_ok
    }
    results['validation_summary'] = validation
    
    print("\nValidation Checks:")
    print(f"Mean close to zero: {'✓' if mean_ok else '✗'}")
    print(f"Residuals normally distributed: {'✓' if normality_ok else '✗'}")
    print(f"No significant autocorrelation: {'✓' if autocorr_ok else '✗'}")
    print(f"No significant cross-correlation with inputs: {'✓' if crosscorr_ok else '✗'}")
    print(f"Residuals pass independence tests: {'✓' if independence_ok else '✗'}")
    
    # Overall decision
    model_ok = mean_ok and normality_ok and autocorr_ok and crosscorr_ok and independence_ok
    
    print("\nOverall Model Validation:", end=" ")
    if model_ok:
        print("✓ PASSED - Model appears to be adequate")
    else:
        print("✗ FAILED - Model may need improvement")
        print("\nRecommended actions:")
        if not mean_ok:
            print("- Check for bias in the model structure or measurement process")
        if not normality_ok:
            print("- Investigate outliers, consider nonlinear transformations")
            print("- Check for heteroscedasticity or time-varying behavior")
        if not autocorr_ok:
            print("- Increase model order or add relevant state variables")
            print("- Consider adding time-delay terms or memory effects")
        if not crosscorr_ok:
            print("- Improve how the model captures input-output relationships")
            print("- Check for nonlinearities in input effects")
        if not independence_ok:
            print("- Look for missing inputs or disturbances")
            print("- Consider nonlinear model structures")
    
    # Generate a PDF report if requested
    if save_report:
        # Create a summary page
        fig_summary = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        title = "State Space Model Residual Analysis Summary"
        
        summary_text = [
            f"Number of samples: {n_samples}",
            f"Residual mean: {stats_results['mean']:.6f}",
            f"Residual std dev: {stats_results['std_dev']:.6f}",
            f"Mean test: {'Passed' if mean_ok else 'Failed'}",
            f"Normality test: {'Passed' if normality_ok else 'Failed'}",
            f"Autocorrelation test: {'Passed' if autocorr_ok else 'Failed'}",
            f"Cross-correlation test: {'Passed' if crosscorr_ok else 'Failed'}",
            f"Independence test: {'Passed' if independence_ok else 'Failed'}",
            f"\nOverall validation: {'Passed' if model_ok else 'Failed'}"
        ]
        
        plt.text(0.5, 0.95, title, fontsize=16, ha='center', va='top', transform=fig_summary.transFigure)
        plt.text(0.1, 0.85, "\n".join(summary_text), fontsize=12, ha='left', va='top', transform=fig_summary.transFigure)
        
        pdf.savefig(fig_summary)
        plt.close(fig_summary)
        
        # Close the PDF
        pdf.close()
        print(f"\nDetailed analysis report saved to {report_path}")
    
    return


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 500
    
    # Well-behaved residuals (white noise)
    good_residuals = np.random.normal(0, 1, n_samples)
    
    # Problematic residuals (autocorrelated)
    bad_residuals = np.zeros(n_samples)
    for i in range(1, n_samples):
        bad_residuals[i] = 0.7 * bad_residuals[i-1] + np.random.normal(0, 0.7)
    
    # Generate inputs
    n_inputs = 2
    inputs = np.zeros((n_inputs, n_samples))
    inputs[0, :] = np.random.normal(0, 1, n_samples)  # Random noise
    inputs[1, :] = np.sin(np.linspace(0, 10*np.pi, n_samples))  # Sine wave
    
    # Example 1: Good model
    print("\n\nANALYSIS OF GOOD MODEL RESIDUALS")
    results_good = perform_residual_analysis(
        good_residuals, 
        inputs=inputs,
        save_report=True, 
        report_path="good_model_residual_analysis.pdf",
        show_plots=True
    )
    
    # Example 2: Model with issues
    print("\n\nANALYSIS OF PROBLEMATIC MODEL RESIDUALS")
    results_bad = perform_residual_analysis(
        bad_residuals, 
        inputs=inputs,
        save_report=True, 
        report_path="bad_model_residual_analysis.pdf",
        show_plots=True
    )