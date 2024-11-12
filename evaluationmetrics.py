import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns

# Function for Distributional Consistency
def calculate_distributional_consistency(real_data, synthetic_data):
    """
    Calculate distributional consistency by comparing statistical moments (mean, variance, skewness)
    of real and synthetic data for each attribute.
    """
    print("=== Distributional Consistency Report ===")
    consistency_report = {}
    attributes = real_data.columns

    for attr in attributes:
        real_mean = np.mean(real_data[attr])
        synthetic_mean = np.mean(synthetic_data[attr])
        real_variance = np.var(real_data[attr])
        synthetic_variance = np.var(synthetic_data[attr])
        real_skewness = real_data[attr].skew()
        synthetic_skewness = synthetic_data[attr].skew()

        print(f"\nAttribute: {attr}")
        print(f"  Real Mean: {real_mean}, Synthetic Mean: {synthetic_mean}")
        print(f"  Real Variance: {real_variance}, Synthetic Variance: {synthetic_variance}")
        print(f"  Real Skewness: {real_skewness}, Synthetic Skewness: {synthetic_skewness}")

        consistency_report[attr] = {
            'real_mean': real_mean,
            'synthetic_mean': synthetic_mean,
            'real_variance': real_variance,
            'synthetic_variance': synthetic_variance,
            'real_skewness': real_skewness,
            'synthetic_skewness': synthetic_skewness
        }
        
    return pd.DataFrame(consistency_report).T

# Function for Statistical Tests
def perform_statistical_tests(real_data, synthetic_data):
    """
    Perform Kolmogorov-Smirnov (K-S) and T-tests for statistical alignment.
    Returns a dictionary of test results.
    """
    print("\n=== Statistical Test Results ===")
    test_results = {}
    attributes = real_data.columns

    for attr in attributes:
        # Kolmogorov-Smirnov test for distributional similarity
        ks_stat, ks_p_value = ks_2samp(real_data[attr], synthetic_data[attr])

        # T-test for mean differences
        t_stat, t_p_value = ttest_ind(real_data[attr], synthetic_data[attr], equal_var=False)

        print(f"\nAttribute: {attr}")
        print(f"  K-S Statistic: {ks_stat}, K-S p-value: {ks_p_value}")
        print(f"  T-Statistic: {t_stat}, T-Test p-value: {t_p_value}")

        test_results[attr] = {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            't_statistic': t_stat,
            't_p_value': t_p_value
        }
    
    return pd.DataFrame(test_results).T

# Function for Qualitative Assessment
def visualize_data_distributions(real_data, synthetic_data, attributes):
    """
    Generate histograms and box plots for qualitative assessment of real vs synthetic data distributions.
    """
    print("\n=== Qualitative Assessment ===")
    for attr in attributes:
        # Histogram comparison
        print(f"\nVisualizing Histogram for {attr}...")
        plt.figure(figsize=(10, 5))
        sns.histplot(real_data[attr], color="blue", label="Real Data", kde=True, stat="density", bins=20)
        sns.histplot(synthetic_data[attr], color="orange", label="Synthetic Data", kde=True, stat="density", bins=20)
        plt.title(f"Histogram Comparison for {attr}")
        plt.xlabel(attr)
        plt.ylabel("Density")
        plt.legend()
        plt.show()

        # Box plot comparison
        print(f"Visualizing Box Plot for {attr}...")
        plt.figure(figsize=(10, 5))
        data = pd.DataFrame({
            f"Real {attr}": real_data[attr],
            f"Synthetic {attr}": synthetic_data[attr]
        })
        sns.boxplot(data=data)
        plt.title(f"Box Plot Comparison for {attr}")
        plt.ylabel(attr)
        plt.show()

# Example usage function to evaluate synthetic data
def evaluate_synthetic_data(real_data, synthetic_data):
    print("\nStarting Evaluation of Synthetic Data...\n")
    
    # Distributional Consistency
    distributional_consistency = calculate_distributional_consistency(real_data, synthetic_data)
    print("\nDistributional Consistency Summary:")
    print(distributional_consistency)

    # Statistical Tests
    statistical_tests = perform_statistical_tests(real_data, synthetic_data)
    print("\nStatistical Test Summary:")
    print(statistical_tests)

    # Qualitative Assessment (Visualizations)
    visualize_data_distributions(real_data, synthetic_data, real_data.columns)

    return distributional_consistency, statistical_tests

# Example of how to run this script
if __name__ == "__main__":
    # Replace the following with actual data
    # Example real and synthetic data (replace with actual datasets)
    real_data = pd.DataFrame({
        'Age': np.random.normal(50, 10, 1000),
        'Risk_Score': np.random.uniform(0, 1, 1000),
        'Lab_Result_1': np.random.normal(100, 15, 1000)
    })
    synthetic_data = pd.DataFrame({
        'Age': np.random.normal(50, 10, 1000),
        'Risk_Score': np.random.uniform(0, 1, 1000),
        'Lab_Result_1': np.random.normal(100, 15, 1000)
    })

    # Run evaluation
    evaluate_synthetic_data(real_data, synthetic_data)
