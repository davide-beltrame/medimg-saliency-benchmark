import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

PLOTS_DIR = "plots"  # Directory containing the performance CSV
EVALUATION_DIR = "evaluation" # Directory containing IoU CSV and for output
PERFORMANCE_CSV_FILENAME = "results.csv" # Name of your performance CSV
IOU_CSV_FILENAME = "saliency_iou_results_intersection_test_0.74.csv" # Expected name of the IoU results CSV

PERFORMANCE_METRICS_TO_CORRELATE = ["accuracy", "precision", "recall", "auroc"]

# Saliency method whose IoU scores will be used for correlation.
# Case-sensitive, must match a column name in your IOU_CSV_FILENAME.
SALIENCY_IOU_COLUMN_FOR_CORRELATION = "Random" 

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

# --- Main Logic ---
def main():
    print(f"--- Performance vs. Agreement Correlation Analysis ---")
    print(f"Using IoU scores from saliency method: {SALIENCY_IOU_COLUMN_FOR_CORRELATION}")

    # 1. Load Performance Data from CSV in PLOTS_DIR
    performance_csv_path = os.path.join(PLOTS_DIR, PERFORMANCE_CSV_FILENAME)
    try:
        perf_df = pd.read_csv(performance_csv_path)
        print(f"Successfully loaded performance data from: {performance_csv_path}")
        # Select only relevant columns (model and the metrics themselves)
        relevant_perf_columns = ['model'] + PERFORMANCE_METRICS_TO_CORRELATE
        # Check if all needed columns exist
        missing_cols = [col for col in relevant_perf_columns if col not in perf_df.columns]
        if missing_cols:
            print(f"Error: Missing performance metric columns in {performance_csv_path}: {missing_cols}")
            return
        
        perf_df = perf_df[relevant_perf_columns]
        # Map short model names to display names
        perf_df['Model'] = perf_df['model'].map(MODEL_NAME_MAPPING)
        if perf_df['Model'].isnull().any():
            print(f"Warning: Some model keys in performance CSV ({perf_df[perf_df['Model'].isnull()]['model'].unique()}) could not be mapped to display names. They will be excluded.")
            perf_df = perf_df.dropna(subset=['Model'])

        # Group by the display model name and average the performance metrics
        # This handles multiple configurations (linear, pretrained) for each base model
        avg_perf_df = perf_df.groupby('Model')[PERFORMANCE_METRICS_TO_CORRELATE].mean().reset_index()
        print("\nAggregated (Averaged) Performance Metrics per Model Type:")
        print(avg_perf_df)

    except FileNotFoundError:
        print(f"Error: Performance CSV file not found at {performance_csv_path}")
        return
    except Exception as e:
        print(f"Error reading or processing performance CSV file {performance_csv_path}: {e}")
        return

    # 2. Load IoU Scores
    iou_csv_path = os.path.join(EVALUATION_DIR, IOU_CSV_FILENAME)
    try:
        iou_df_raw = pd.read_csv(iou_csv_path)
        # Assuming the first column is the model name and should be the index
        iou_df = iou_df_raw.set_index(iou_df_raw.columns[0])
        if SALIENCY_IOU_COLUMN_FOR_CORRELATION not in iou_df.columns:
            print(f"Error: Saliency method '{SALIENCY_IOU_COLUMN_FOR_CORRELATION}' not found as a column in {iou_csv_path}.")
            print(f"Available columns: {list(iou_df.columns)}")
            return
        # Select only the relevant IoU column
        iou_scores = iou_df[[SALIENCY_IOU_COLUMN_FOR_CORRELATION]].copy()
        iou_scores.rename(columns={SALIENCY_IOU_COLUMN_FOR_CORRELATION: 'IoU'}, inplace=True)
        print(f"\nSuccessfully loaded IoU data from {iou_csv_path} for column '{SALIENCY_IOU_COLUMN_FOR_CORRELATION}'")
        print(iou_scores)

    except FileNotFoundError:
        print(f"Error: IoU CSV file not found at {iou_csv_path}")
        return
    except Exception as e:
        print(f"Error reading IoU CSV file {iou_csv_path}: {e}")
        return

    # 3. Merge Performance and IoU data
    # The index of iou_scores is 'Model' (display name), avg_perf_df has a 'Model' column
    combined_df = pd.merge(avg_perf_df, iou_scores, on="Model", how="inner")
    
    if combined_df.empty:
        print("\nError: Combined data is empty. Ensure model names match between performance and IoU data.")
        print(f"  Models from performance data: {avg_perf_df['Model'].unique()}")
        print(f"  Models from IoU data: {iou_scores.index.unique()}")
        return
    
    # Drop rows with any NaN values that might prevent correlation (e.g., if a model had NaN IoU)
    combined_df_cleaned = combined_df.dropna()

    if len(combined_df_cleaned) < 2:
        print(f"\nError: Need at least two models with complete data (all performance metrics and specified IoU) after merging and cleaning. Found {len(combined_df_cleaned)}. Exiting.")
        print("Combined data before dropping NaN:")
        print(combined_df)
        print("Combined data after dropping NaN:")
        print(combined_df_cleaned)
        return

    print(f"\nModels included in correlation analysis ({len(combined_df_cleaned)}): {combined_df_cleaned['Model'].tolist()}")
    print("Data used for correlation:")
    print(combined_df_cleaned)


    # 4. Calculate Correlations
    correlation_results_list = []
    for perf_metric in PERFORMANCE_METRICS_TO_CORRELATE:
        if perf_metric not in combined_df_cleaned.columns:
            print(f"  Warning: Performance metric '{perf_metric}' not found in combined data. Skipping.")
            continue

        performance_values = combined_df_cleaned[perf_metric].values
        iou_values = combined_df_cleaned['IoU'].values

        if len(performance_values) < 2: # Should be caught by earlier check on combined_df_cleaned
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