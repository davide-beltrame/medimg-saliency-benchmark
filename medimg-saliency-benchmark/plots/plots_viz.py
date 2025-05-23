import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Specify the CSV file path
CSV_FILE = "results.csv"  

# Read and process the CSV file with the new format
def read_csv_and_process(csv_file_path):
    """
    Read the CSV file with structure:
    model,linear,pretrained,accuracy,pm,precision,pm,specificity,pm,recall,pm,f1,pm,auroc,pm
    """
    # Read the raw CSV
    df = pd.read_csv(csv_file_path)
    
    # Create a clean dataframe with proper structure
    processed_data = []
    
    # Calculate the number of metrics (excluding the first 3 columns and accounting for the pm columns)
    num_columns = len(df.columns)
    num_metrics = (num_columns - 3) // 2  # Each metric has a value and a pm column
    
    # Process each row
    for _, row in df.iterrows():
        model = row[0]  # model
        linear = row[1]  # linear
        pretrained = row[2]  # pretrained
        model_name = f"{model}_{linear}_{pretrained}"
        if not linear or not pretrained:
            continue
        
        # Extract metric values and their confidence intervals
        metrics_data = {}
        metrics_index = 3  # Start from the 4th column (0-indexed)
        
        # Expected metrics in order: accuracy, precision, specificity, recall, f1, auroc
        metric_names = ["accuracy", "precision", "specificity", "recall", "f1", "auroc"]
        
        for i in range(num_metrics):
            if metrics_index < len(row):
                metric_name = metric_names[i] if i < len(metric_names) else f"metric_{i}"
                metrics_data[f"{metric_name}_mean"] = row[metrics_index]
                metrics_data[f"{metric_name}_ci_half"] = row[metrics_index + 1]  # pm value
                metrics_index += 2  # Move to the next metric
        
        # Create a row for the model
        processed_row = {
            'model': model,
            'linear': linear,
            'pretrained': pretrained,
            'model_name': model_name,
            **metrics_data  # Add all metrics data
        }
        
        processed_data.append(processed_row)
    
    return pd.DataFrame(processed_data)

# Create bar plots with confidence intervals for each metric
def plot_metrics(df):
    # List of metrics to plot
    metrics = ["accuracy", "precision", "specificity", "recall", "f1", "auroc"]
    
    # Set up the figure size and style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Create a directory for plots if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    for metric in metrics:
        # Skip metrics that don't exist in the dataframe
        if f"{metric}_mean" not in df.columns:
            print(f"Skipping {metric} as it's not in the data")
            continue
            
        plt.figure()
        
        # Prepare data for plotting
        model_names = df['model_name']
        means = df[f'{metric}_mean']
        errors = df[f'{metric}_ci_half']
        
        # Create bar plot with error bars
        plt.bar(model_names, means, yerr=errors, capsize=10, color='skyblue', edgecolor='black')
        
        # Customize the plot
        plt.title(f'{metric.upper()} with 95% Confidence Intervals', fontsize=16)
        plt.xlabel('Models', fontsize=14)
        plt.ylabel(f'{metric.capitalize()}', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.savefig(f'plots/{metric}_barplot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Generated plots for available metrics")

# Create a single plot with all metrics using seaborn
def plot_all_metrics_together(df):
    # Get all metric columns
    metric_columns = [col for col in df.columns if col.endswith('_mean')]
    ci_columns = [col for col in df.columns if col.endswith('_ci_half')]
    
    # Create a long-form dataframe for seaborn
    melted_df = pd.melt(df, id_vars=['model_name'], 
                        value_vars=metric_columns,
                        var_name='metric', value_name='value')
    
    # Add CI information
    melted_df['metric_base'] = melted_df['metric'].str.replace('_mean', '')
    melted_ci = pd.melt(df, id_vars=['model_name'], 
                       value_vars=ci_columns,
                       var_name='ci_metric', value_name='ci')
    melted_ci['metric_base'] = melted_ci['ci_metric'].str.replace('_ci_half', '')
    
    # Merge the dataframes
    final_df = pd.merge(melted_df, melted_ci, on=['model_name', 'metric_base'])
    
    # Clean up the metric names for display
    final_df['clean_metric'] = final_df['metric'].str.replace('_mean', '').str.capitalize()
    
    # Create the plot
    plt.figure(figsize=(18, 10))
    
    # Plot grouped bar chart
    ax = sns.barplot(x='clean_metric', y='value', hue='model_name', data=final_df)
    
    # Add error bars
    x_coords = np.arange(len(final_df['clean_metric'].unique()))
    width = 0.8 / len(df['model_name'].unique())
    
    for i, model in enumerate(df['model_name'].unique()):
        model_data = final_df[final_df['model_name'] == model]
        x_pos = x_coords - 0.4 + width * (i + 0.5)
        plt.errorbar(x=x_pos, y=model_data['value'], yerr=model_data['ci'], 
                    fmt='none', capsize=5, ecolor='black', alpha=0.75)
    
    # Customize plot
    plt.title('Performance Metrics Comparison', fontsize=16)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('plots/all_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated combined metrics plot")


# Create plots directory
os.makedirs("plots", exist_ok=True)

try:
    # Process the CSV file
    df = read_csv_and_process(CSV_FILE)

    # Create individual plots for each metric
    plot_metrics(df)
    
    # Create a combined plot with all metrics
    plot_all_metrics_together(df)
    
except FileNotFoundError:
    raise Exception(f"CSV file '{CSV_FILE}' not found. Please make sure the file exists.")