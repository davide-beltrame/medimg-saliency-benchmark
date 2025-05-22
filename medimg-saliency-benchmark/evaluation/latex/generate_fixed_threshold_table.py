#!/usr/bin/env python3
"""
Script to generate a LaTeX table of model-expert agreement results using a fixed threshold of 0.5.
"""
import os

# Define paths
EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(EVALUATION_DIR, "model_expert_agreement_fixed_threshold_table.tex")

# Manually define the model results based on provided data with fixed threshold of 0.5
# Format: [model_name, 
#          [gradcam_thr, gradcam_iou, gradcam_iou_pval, gradcam_pg, gradcam_pg_pval], 
#          [cam_thr, cam_iou, cam_iou_pval, cam_pg, cam_pg_pval], 
#          [rand_thr, rand_iou, rand_pg]]
model_results = [
    ["AlexNet", 
     [0.50, 9.08, 0.0002, 28.00, 0.0000], 
     [0.50, 6.78, 0.7558, 12.00, 0.0061],
     [0.50, 5.93, 0.00]],
    
    ["InceptionNet-V1", 
     [0.50, 16.33, 0.0000, 38.00, 0.0000], 
     [0.50, 16.33, 0.0000, 38.00, 0.0000],
     [0.50, 5.36, 0.00]],
    
    ["ResNet-50", 
     [0.50, 11.46, 0.0000, 20.00, 0.0005], 
     [0.50, 10.84, 0.0000, 20.00, 0.0005],
     [0.50, 4.99, 0.00]],
    
    ["VGG-16", 
     [0.50, 19.60, 0.0000, 32.00, 0.0000], 
     [0.50, 22.19, 0.0000, 40.00, 0.0000],
     [0.50, 4.96, 0.00]],
]

# Find max values for bolding
max_iou = 22.19  # VGG-16 CAM
max_pg = 40.00   # VGG-16 CAM

# Generate the LaTeX table
table = r"""
\begin{table*}[t]
    \centering
    \setlength{\tabcolsep}{3pt}
    \caption{\textbf{Model-Expert Agreement with Fixed Threshold (0.5).} Entries show IoU and PG between model-generated saliency maps and expert annotations, along with p-values.}
    \begin{tabularx}{\linewidth}{l|YY|YY|YY}
        \toprule
        \multirow{2}{*}{Model} & \multicolumn{2}{c|}{GradCAM} & \multicolumn{2}{c|}{CAM} & \multicolumn{2}{c}{Random} \\
        & IoU & PG & IoU & PG & IoU & PG \\
        \midrule
"""

# Add rows for each model
for model in model_results:
    model_name = model[0]
    gradcam_data = model[1]  # [thr, iou, iou_pval, pg, pg_pval]
    cam_data = model[2]      # [thr, iou, iou_pval, pg, pg_pval]
    random_data = model[3]   # [thr, iou, pg]
    
    # Format values with bold for max values
    gradcam_iou = f"{gradcam_data[1]}"
    gradcam_pg = f"{gradcam_data[3]}"
    cam_iou = f"{cam_data[1]}"
    cam_pg = f"{cam_data[3]}"
    
    # Add bold and underline formatting to max values
    if gradcam_data[1] == max_iou:
        gradcam_iou = r"\textbf{\underline{" + gradcam_iou + r"}}"
    if gradcam_data[3] == max_pg:
        gradcam_pg = r"\textbf{\underline{" + gradcam_pg + r"}}"
    if cam_data[1] == max_iou:
        cam_iou = r"\textbf{\underline{" + cam_iou + r"}}"
    if cam_data[3] == max_pg:
        cam_pg = r"\textbf{\underline{" + cam_pg + r"}}"
    
    # Format p-values
    gradcam_iou_pval = f"{{\\scriptsize {gradcam_data[2]:.3f}}}"
    gradcam_pg_pval = f"{{\\scriptsize {gradcam_data[4]:.3f}}}"
    cam_iou_pval = f"{{\\scriptsize {cam_data[2]:.3f}}}"
    cam_pg_pval = f"{{\\scriptsize {cam_data[4]:.3f}}}"
    
    # Row for this model
    row = f"        {model_name} & {gradcam_iou} {gradcam_iou_pval} & {gradcam_pg} {gradcam_pg_pval} & {cam_iou} {cam_iou_pval} & {cam_pg} {cam_pg_pval} & {random_data[1]} {{\\scriptsize -}} & {random_data[2]} {{\\scriptsize -}} \\\\"
    table += row + "\n"

# Complete the table
table += r"""        \bottomrule
    \end{tabularx}
    \label{tb:ModelSaliencyComparisonFixedThreshold}
\end{table*}
"""

# Write to file
with open(OUTPUT_PATH, 'w') as f:
    f.write(table)

print(f"LaTeX table generated and saved to {OUTPUT_PATH}")

# Print the table to console for review
print("\nGenerated LaTeX Table:")
print(table)
