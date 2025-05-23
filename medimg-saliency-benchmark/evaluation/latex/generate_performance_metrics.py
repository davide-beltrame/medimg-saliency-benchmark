import pandas as pd

# Read the CSV file
df = pd.read_csv('plots/results.csv')

# Process the data to restructure it
# Map "linear" column to "Adapted Classifier" for the table
df['adapted_classifier'] = df['linear'].map({True: 'Yes', False: 'No'})
df['pretrained_text'] = df['pretrained'].map({True: 'Yes', False: 'No'})

# Format metrics with their plus/minus values
metrics = ['accuracy', 'f1', 'auroc', 'specificity']  # Changed to use accuracy and f1 instead of precision and recall
for i, row in df.iterrows():
    for metric in metrics:
        # Assuming the pm column follows each metric column
        metric_idx = df.columns.get_loc(metric)
        pm_idx = metric_idx + 1
        
        # Format the value with its plus/minus
        if pd.notna(row[metric]) and pd.notna(row[df.columns[pm_idx]]):
            df.at[i, f"{metric}_formatted"] = f"{row[metric]:.2f} $\\pm$ {row[df.columns[pm_idx]]:.2f}"
        else:
            df.at[i, f"{metric}_formatted"] = "- $\\pm$ -"

# Generate LaTeX table
latex_table = """\\begin{table*}[t]
    \\centering
    \\setlength{\\tabcolsep}{3pt}
    \\caption{Model Performance Metrics}
    \\begin{tabularx}{\\textwidth}{lYYYcccc}
        \\toprule
        Model & Pretrained & Adapted Classifier & Accuracy & F1 & ROC AUC & Specifity\\\\
        \\midrule
"""

# Group by model to handle the multirow structure
models = df['model'].unique()
mapped_names = {
    "vgg": "VGG-16",
    "in": "InceptionNet-V1",
    "an": "AlexNet",
    "rn": "ResNet-50",
}
for i, model in enumerate(models):
    model_df = df[df['model'] == model]
    
    # Group by pretrained status
    pretrained_values = model_df['pretrained'].unique()
    
    # Count rows for multirow
    model_rows = len(model_df)
    model_name = mapped_names[model]
    for j, pretrained in enumerate(pretrained_values):
        pretrained_df = model_df[model_df['pretrained'] == pretrained]
        pretrained_rows = len(pretrained_df)
        
        # Add pretrained status with multirow
        for k, (idx, row) in enumerate(pretrained_df.iterrows()):
            if k == 0 and j == 0:
                # First row of the model
                latex_table += f"        \\multirow{{{model_rows}}}{{*}}{{{model_name}}} &  \\multirow{{{pretrained_rows}}}{{*}}{{{row['pretrained_text']}}}"
            elif k == 0:
                # First row of a new pretrained group
                latex_table += f"         &  \\multirow{{{pretrained_rows}}}{{*}}{{{row['pretrained_text']}}}"
            else:
                # Continuation rows
                latex_table += "         &                          "

            latex_table += f" &   {row['adapted_classifier'] if model not in {'in', 'rn'} else '-'}  &   {row['accuracy_formatted']}     &   {row['f1_formatted']}    &   {row['auroc_formatted']} &   {row['specificity_formatted']}\\\\\n"
        
        # Add cmidrule between pretrained groups within the same model (except after the last pretrained group)
        if j < len(pretrained_values) - 1:
            latex_table += "        \\cmidrule(lr){2-6}\n"
    
    # Add midrule between models (except after the last model)
    if i < len(models) - 1:
        latex_table += "        \\midrule\n"

# Close the table
latex_table += """        \\bottomrule
    \\end{tabularx}
    \\label{tab:Performance}
\\end{table*}
"""

print(latex_table)
# Optionally save to a file
with open('plots/results.tex', 'w') as f:
    f.write(latex_table)