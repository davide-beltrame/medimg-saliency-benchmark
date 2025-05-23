import csv

PATH_TO_CSV = "evaluation/model-expert-agreement.csv"

mapped_names = {
    "vgg": "VGG-16",
    "in": "InceptionNet-V1",
    "an": "AlexNet",
    "rn": "ResNet-50",
}

def bold_if_significant(value, pval):
    """Bold the value if p-value < 0.05"""
    value = f"{float(value)*100:.2f}"
    try:
        pval = f"{float(pval):.3f}"
        if float(pval) > 0.05:
            return f"\\textit{{{value} {{\\scriptsize {pval}}}}}"
        else:
            return f"{value} {{\\scriptsize {pval}}}"
    except (ValueError, TypeError):
        return f"{value} {{\\scriptsize {pval}}}"

def create_latex_table(csv_file):
    # Read the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader if row.get('linear').lower() == 'true' and row.get('pretrained').lower() == 'true']
    
    if not data:
        print("No models with linear=true and pretrained=true found in the CSV.")
        return
    
    # Start creating the LaTeX table
    latex_table = []
    latex_table.append("\\begin{table*}[t]")
    latex_table.append("    \\centering")
    latex_table.append("    \\setlength{\\tabcolsep}{3pt}")
    latex_table.append("    \\caption{\\textbf{Model-Expert Agreement.} Entries show the degree of agreement between model-generated saliency maps and expert annotations, measured by Intersection over Union (IoU) and Pointing Game (PG) metrics. P-values are reported in smaller font, and the \\underline{non}-statistically significant results are highlighted in italic.}")
    
    # Create the tabular environment with multicolumns
    latex_table.append("    \\begin{tabularx}{\\linewidth}{l|YY|YY|YY}")
    latex_table.append("        \\toprule")
    latex_table.append("        \\multirow{2}{*}{Model} & \\multicolumn{2}{c|}{GradCAM} & \\multicolumn{2}{c|}{CAM} & \\multicolumn{2}{c}{Random} \\\\")
    latex_table.append("        & IoU & PG & IoU & PG & IoU & PG \\\\")
    latex_table.append("        \\midrule")
    
    # Add each model as a row
    for row in data:
        model_name = row.get('model', 'Unknown')
        model_name = mapped_names[model_name]

        # Format each cell with value and p-value in scriptsize
        gradcam_iou = bold_if_significant(row.get('GradCAM_iou', '-'), row.get('GradCAM_iou_pval', '-'))
        gradcam_pg = bold_if_significant(row.get('GradCAM_pg', '-'), row.get('GradCAM_pg_pval', '-'))
        
        cam_iou = bold_if_significant(row.get('CAM_iou', '-'), row.get('CAM_iou_pval', '-'))
        cam_pg = bold_if_significant(row.get('CAM_pg', '-'), row.get('CAM_pg_pval', '-'))
        
        random_iou = bold_if_significant(row.get('Random_iou', '-'), row.get('Random_iou_pval', '-'))
        random_pg = bold_if_significant(row.get('Random_pg', '-'), row.get('Random_pg_pval', '-'))
        
        # Add the row
        latex_table.append(f"        {model_name} & {gradcam_iou} & {gradcam_pg} & {cam_iou} & {cam_pg} & {random_iou} & {random_pg} \\\\")
    
    # Finish the table
    latex_table.append("        \\bottomrule")
    latex_table.append("    \\end{tabularx}")
    latex_table.append("    \\label{tb:ModelSaliencyComparison}")
    latex_table.append("\\end{table*}")
    
    # Join all lines and return
    return "\n".join(latex_table)

latex = create_latex_table(PATH_TO_CSV)
with open(PATH_TO_CSV.replace(".csv", ".txt"), "w") as f:
    f.write(latex)
print(latex)