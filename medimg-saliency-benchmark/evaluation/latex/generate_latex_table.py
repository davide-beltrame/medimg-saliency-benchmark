#!/usr/bin/env python3
"""
Script to generate a LaTeX table from model agreement CSV files.
"""
import os
import pandas as pd

# Map model short names to full names
MODEL_NAME_MAP = {
    "an": "AlexNet",
    "in": "InceptionNet-V1",
    "rn": "ResNet-50",
    "vgg": "VGG-16"
}

def format_value(value, pval=None, is_random=False, is_best=False):
    """Format value for LaTeX table with optional p-value and/or bold formatting"""
    # Multiply value by 100 to convert from decimal to percentage
    value = value * 100
    
    # Format the value with 2 decimal places
    formatted = f"{value:.2f}"
    
    # Bold the best results
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    
    # Add p-value in scriptsize if provided
    if is_random:
        return f"{formatted} {{\\scriptsize -}}"
    elif pval is not None:
        pval_str = f"{{\\scriptsize {pval:.3f}}}"
        return f"{formatted} {pval_str}"
    
    return formatted

def main():
    # Read the three CSV files
    gradcam_df = pd.read_csv("model-expert-agreement-gradcam.csv")
    cam_df = pd.read_csv("model-expert-agreement-cam.csv")
    random_df = pd.read_csv("model-expert-agreement-random.csv")
    
    # Create the LaTeX table
    latex_table = []
    latex_table.append("\\begin{table*}[t]")
    latex_table.append("    \\centering")
    latex_table.append("    \\setlength{\\tabcolsep}{2pt}")  # Make columns a bit narrower
    latex_table.append("    \\caption{\\textbf{Model-Expert Agreement.} Entries show the degree of agreement between model-generated saliency maps and expert annotations, measured by Intersection over Union (IoU) and Pointing Game (PG) metrics. P-values are reported in smaller font. Best results for each model are shown in \\textbf{bold}.}")
    latex_table.append("    \\begin{tabularx}{\\linewidth}{l|cYY|cYY|cYY}")  # Use 'c' for narrower threshold columns
    latex_table.append("        \\toprule")
    latex_table.append("        \\multirow{2}{*}{Model} & \\multicolumn{3}{c|}{GradCAM} & \\multicolumn{3}{c|}{CAM} & \\multicolumn{3}{c}{Random} \\\\")
    latex_table.append("        & Thr & IoU & PG & Thr & IoU & PG & Thr & IoU & PG \\\\")
    latex_table.append("        \\midrule")
    
    # Process each model
    for model_key in MODEL_NAME_MAP.keys():
        # Get data for each model
        gradcam_row = gradcam_df[gradcam_df['model'] == model_key].iloc[0]
        cam_row = cam_df[cam_df['model'] == model_key].iloc[0]
        random_row = random_df[random_df['model'] == model_key].iloc[0]
        
        # Format model name
        model_name = MODEL_NAME_MAP[model_key]
        
        # Determine best values for this model
        best_iou = max(gradcam_row['iou'], cam_row['iou'], random_row['iou'])
        best_pg = max(gradcam_row['pg'], cam_row['pg'], random_row['pg'])
        
        # Format thresholds
        gradcam_thr = f"{{\\scriptsize {gradcam_row['threshold']:.2f}}}"
        cam_thr = f"{{\\scriptsize {cam_row['threshold']:.2f}}}"
        random_thr = f"{{\\scriptsize {random_row['threshold']:.2f}}}"
        
        # Format IoU and PG values with p-values and bold for best
        gradcam_iou = format_value(gradcam_row['iou'], gradcam_row['iou_pval'], is_best=(gradcam_row['iou'] == best_iou))
        gradcam_pg = format_value(gradcam_row['pg'], gradcam_row['pg_pval'], is_best=(gradcam_row['pg'] == best_pg))
        cam_iou = format_value(cam_row['iou'], cam_row['iou_pval'], is_best=(cam_row['iou'] == best_iou))
        cam_pg = format_value(cam_row['pg'], cam_row['pg_pval'], is_best=(cam_row['pg'] == best_pg))
        random_iou = format_value(random_row['iou'], is_random=True, is_best=(random_row['iou'] == best_iou))
        random_pg = format_value(random_row['pg'], is_random=True, is_best=(random_row['pg'] == best_pg))
        
        # Add row to table
        table_row = f"        {model_name} & {gradcam_thr} & {gradcam_iou} & {gradcam_pg} & {cam_thr} & {cam_iou} & {cam_pg} & {random_thr} & {random_iou} & {random_pg} \\\\"
        latex_table.append(table_row)
    
    # Finish the table
    latex_table.append("        \\bottomrule")
    latex_table.append("    \\end{tabularx}")
    latex_table.append("    \\label{tb:ModelSaliencyComparison}")
    latex_table.append("\\end{table*}")
    
    # Join table lines and write to file
    latex_content = "\n".join(latex_table)
    with open("model_expert_agreement_table.tex", "w") as f:
        f.write(latex_content)
    
    print("LaTeX table generated as 'model_expert_agreement_table.tex'")
    print("\nTable Preview:")
    print(latex_content)

if __name__ == "__main__":
    main() 