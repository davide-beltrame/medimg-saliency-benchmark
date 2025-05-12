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

def format_value(value, pval=None, is_random=False):
    """Format value for LaTeX table with optional p-value"""
    # Multiply value by 100 to convert from decimal to percentage
    value = value * 100
    
    if is_random:
        return f"{value:.2f} {{\\scriptsize -}}"
    
    # Format the value with 2 decimal places
    formatted = f"{value:.2f}"
    
    # Add p-value in scriptsize if provided
    if pval is not None:
        # Determine if statistically significant (p < 0.05)
        is_significant = pval < 0.05
        pval_str = f"{{\\scriptsize {pval:.3f}}}"
        
        # If not significant, wrap in \textit{}
        if not is_significant:
            return f"\\textit{{{formatted} {pval_str}}}"
        else:
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
    latex_table.append("    \\setlength{\\tabcolsep}{3pt}")
    latex_table.append("    \\caption{\\textbf{Model-Expert Agreement.} Entries show the degree of agreement between model-generated saliency maps and expert annotations, measured by Intersection over Union (IoU) and Pointing Game (PG) metrics. P-values are reported in smaller font, and the \\underline{non}-statistically significant results are highlighted in italic. Threshold values used for each method are shown in the column headers or as separate columns when they vary by model.}")
    latex_table.append("    \\begin{tabularx}{\\linewidth}{l|YY|YYY|YY}")
    latex_table.append("        \\toprule")
    latex_table.append("        \\multirow{2}{*}{Model} & \\multicolumn{2}{c|}{GradCAM (thr=0.74)} & \\multicolumn{3}{c|}{CAM} & \\multicolumn{2}{c}{Random (thr=0.50)} \\\\")
    latex_table.append("        & IoU & PG & Thr & IoU & PG & IoU & PG \\\\")
    latex_table.append("        \\midrule")
    
    # Process each model
    for model_key in MODEL_NAME_MAP.keys():
        # Get data for each model
        gradcam_row = gradcam_df[gradcam_df['model'] == model_key].iloc[0]
        cam_row = cam_df[cam_df['model'] == model_key].iloc[0]
        random_row = random_df[random_df['model'] == model_key].iloc[0]
        
        # Format model name
        model_name = MODEL_NAME_MAP[model_key]
        
        # Format thresholds (only needed for CAM as it varies)
        cam_thr = f"{{\\scriptsize {cam_row['threshold']:.2f}}}"
        
        # Format IoU and PG values with p-values
        gradcam_iou = format_value(gradcam_row['iou'], gradcam_row['iou_pval'])
        gradcam_pg = format_value(gradcam_row['pg'], gradcam_row['pg_pval'])
        cam_iou = format_value(cam_row['iou'], cam_row['iou_pval'])
        cam_pg = format_value(cam_row['pg'], cam_row['pg_pval'])
        random_iou = format_value(random_row['iou'], is_random=True)
        random_pg = format_value(random_row['pg'], is_random=True)
        
        # Add row to table
        table_row = f"        {model_name} & {gradcam_iou} & {gradcam_pg} & {cam_thr} & {cam_iou} & {cam_pg} & {random_iou} & {random_pg} \\\\"
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