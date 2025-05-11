import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

PLOTS_DIR = "plots"  # Directory containing the performance CSV
EVALUATION_DIR = "evaluation" # Directory containing IoU CSV and for output
PERFORMANCE_CSV_FILENAME = "results.csv" # Name of your performance CSV
IOU_CSV_FILENAME = "saliency_iou_results_full_test_0.74.csv" # Expected name of the IoU results CSV

PERFORMANCE_METRICS_TO_CORRELATE = ["accuracy", "precision", "recall", "auroc"]

# Saliency method whose IoU scores will be used for correlation.
SALIENCY_IOU_COLUMN_FOR_CORRELATION = "GradCAM" # "CAM", "GradCAM", "Random"

# Mapping from 'model' column in performance CSV to display names used in IoU CSV and final output
MODEL_NAME_MAPPING = {
    "an": "AlexNet",
    "in": "InceptionNet",
    "rn": "ResNet",
    "vgg": "VGG"
}
# Inverse mapping for convenience if needed, though not used in current logic directly
DISPLAY_NAME_TO_PREFIX_MAPPING = {v: k for k, v in MODEL_NAME_MAPPING.items()}

OUTPUT_CORRELATION_CSV_FILENAME = f"correlation_performance_iou_{SALIENCY_IOU_COLUMN_FOR_CORRELATION}.csv"

def main():
    print(f"--- Performance vs. Agreement Correlation Analysis ---")
    print(f"Using IoU scores from saliency method: {SALIENCY_IOU_COLUMN_FOR_CORRELATION}")

    # 1. Load Performance Data from CSV in PLOTS_DIR
    performance_csv_path = os.path.join(PLOTS_DIR, PERFORMANCE_CSV_FILENAME)
    try:
        perf_df = pd.read_csv(performance_csv_path)
        print(f"Successfully loaded performance data from: {performance_csv_path}")
        # Select only relevant columns (model, linear, pretrained, and the metrics themselves)
        relevant_perf_columns = ['model', 'linear', 'pretrained'] + PERFORMANCE_METRICS_TO_CORRELATE
        # Check if all needed columns exist
        missing_cols = [col for col in relevant_perf_columns if col not in perf_df.columns]
        if missing_cols:
            print(f"Error: Missing performance metric columns in {performance_csv_path}: {missing_cols}")
            return
        
        perf_df = perf_df[relevant_perf_columns]
        print("\nPerformance data (all configurations):")
        print(perf_df)

    except FileNotFoundError:
        print(f"Error: Performance CSV file not found at {performance_csv_path}")
        return
    except Exception as e:
        print(f"Error reading or processing performance CSV file {performance_csv_path}: {e}")
        return

    # 2. Load IoU Scores
    iou_csv_path = os.path.join(EVALUATION_DIR, IOU_CSV_FILENAME)
    try:
        iou_df = pd.read_csv(iou_csv_path)
        
        if SALIENCY_IOU_COLUMN_FOR_CORRELATION not in iou_df.columns:
            print(f"Error: Saliency method '{SALIENCY_IOU_COLUMN_FOR_CORRELATION}' not found as a column in {iou_csv_path}.")
            print(f"Available columns: {list(iou_df.columns)}")
            return
        
        # Select only relevant columns for the comparison
        iou_df = iou_df[['model', 'linear', 'pretrained', SALIENCY_IOU_COLUMN_FOR_CORRELATION]]
        iou_df.rename(columns={SALIENCY_IOU_COLUMN_FOR_CORRELATION: 'IoU'}, inplace=True)
            
        print(f"\nSuccessfully loaded IoU data from {iou_csv_path} for column '{SALIENCY_IOU_COLUMN_FOR_CORRELATION}'")
        print(iou_df)

    except FileNotFoundError:
        print(f"Error: IoU CSV file not found at {iou_csv_path}")
        return
    except Exception as e:
        print(f"Error reading IoU CSV file {iou_csv_path}: {e}")
        return

    # 3. Merge Performance and IoU data on model, linear, and pretrained
    combined_df = pd.merge(perf_df, iou_df, on=['model', 'linear', 'pretrained'], how="inner")
    
    if combined_df.empty:
        print("\nError: Combined data is empty. Ensure configurations match between performance and IoU data.")
        return
    
    # Drop rows with any NaN values that might prevent correlation
    combined_df_cleaned = combined_df.dropna()

    if len(combined_df_cleaned) < 2:
        print(f"\nError: Need at least two configurations with complete data. Found {len(combined_df_cleaned)}. Exiting.")
        return

    print(f"\nData points for correlation analysis ({len(combined_df_cleaned)}):")
    print(combined_df_cleaned)

    # Add model display names for easier interpretation
    combined_df_cleaned['Model_Name'] = combined_df_cleaned['model'].map(MODEL_NAME_MAPPING)
    
    # 4. Calculate Correlations
    correlation_results_list = []
    for perf_metric in PERFORMANCE_METRICS_TO_CORRELATE:
        if perf_metric not in combined_df_cleaned.columns:
            print(f"  Warning: Performance metric '{perf_metric}' not found in combined data. Skipping.")
            continue

        performance_values = combined_df_cleaned[perf_metric].values
        iou_values = combined_df_cleaned['IoU'].values

        if len(performance_values) < 2:
            print(f"  Skipping correlation for '{perf_metric}': Not enough data points.")
            pearson_val, spearman_val = np.nan, np.nan
            pearson_p, spearman_p = np.nan, np.nan
        else:
            pearson_val, pearson_p = pearsonr(performance_values, iou_values)
            spearman_val, spearman_p = spearmanr(performance_values, iou_values)
        
        correlation_results_list.append({
            "Performance Metric": perf_metric.capitalize(),
            "Pearson Correlation": pearson_val,
            "Pearson p-value": pearson_p,
            "Spearman Correlation": spearman_val,
            "Spearman p-value": spearman_p,
            "N": len(performance_values)
        })

    # 5. Display and Save Results
    output_correlation_df = pd.DataFrame(correlation_results_list)
    
    print("\n--- Performance-Agreement Correlation Results ---")
    def format_corr_for_display(row):
        if pd.isna(row["Pearson Correlation"]) or pd.isna(row["Spearman Correlation"]):
            return "NaN"
        # Format as (Pearson, Spearman)
        return f"({row['Pearson Correlation']:.3f}, {row['Spearman Correlation']:.3f})"

    output_correlation_df["Correlation with IoU"] = output_correlation_df.apply(format_corr_for_display, axis=1)
    display_table_df = output_correlation_df[["Performance Metric", "Correlation with IoU", "N"]]
    
    print(display_table_df.to_string(index=False))
    
    # Print p-values separately for analysis
    print("\nDetailed P-value analysis:")
    p_value_df = output_correlation_df[["Performance Metric", "Pearson p-value", "Spearman p-value"]]
    print(p_value_df.to_string(index=False))

    # Generate a scatter plot to visualize the correlations
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(PERFORMANCE_METRICS_TO_CORRELATE):
            plt.subplot(2, 2, i+1)
            plt.scatter(combined_df_cleaned[metric], combined_df_cleaned['IoU'])
            
            # Add labels for each point
            for j, row in combined_df_cleaned.iterrows():
                label = f"{row['Model_Name']}\n{row['linear']}-{row['pretrained']}"
                plt.annotate(label, (row[metric], row['IoU']), fontsize=7)
            
            # Add trend line
            z = np.polyfit(combined_df_cleaned[metric], combined_df_cleaned['IoU'], 1)
            p = np.poly1d(z)
            plt.plot(combined_df_cleaned[metric], p(combined_df_cleaned[metric]), "r--", alpha=0.7)
            
            # Add correlation coefficient and p-value
            corr_row = output_correlation_df[output_correlation_df["Performance Metric"] == metric.capitalize()].iloc[0]
            plt.title(f"{metric.capitalize()} vs IoU\nPearson r={corr_row['Pearson Correlation']:.3f} (p={corr_row['Pearson p-value']:.3f})")
            plt.xlabel(metric.capitalize())
            plt.ylabel(f"IoU ({SALIENCY_IOU_COLUMN_FOR_CORRELATION})")
        
        plt.tight_layout()
        plot_path = os.path.join(EVALUATION_DIR, f"correlation_plot_{SALIENCY_IOU_COLUMN_FOR_CORRELATION}.png")
        plt.savefig(plot_path)
        print(f"\nCorrelation plot saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating correlation plot: {e}")

    # Save the detailed results to CSV
    output_path = os.path.join(EVALUATION_DIR, OUTPUT_CORRELATION_CSV_FILENAME)
    try:
        os.makedirs(EVALUATION_DIR, exist_ok=True) # Ensure evaluation directory exists
        output_correlation_df.to_csv(output_path, index=False, float_format="%.4f")
        print(f"\nCorrelation results saved to: {output_path}")
    except Exception as e:
        print(f"\nError saving correlation results to CSV: {e}")

if __name__ == "__main__":
    main()