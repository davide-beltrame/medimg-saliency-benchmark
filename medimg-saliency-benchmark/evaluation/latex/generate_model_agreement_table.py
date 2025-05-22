#!/usr/bin/env python3
"""
Script to generate a LaTeX table of model-expert agreement results using manually specified values.
"""
import os

# Define paths
EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(EVALUATION_DIR, "model_expert_agreement_table.tex")

# Manually define the model results based on provided data
# Format: [model_name, 
#          [gradcam_thr, gradcam_iou, gradcam_iou_pval, gradcam_pg, gradcam_pg_pval], 
#          [cam_thr, cam_iou, cam_iou_pval, cam_pg, cam_pg_pval], 
#          [rand_thr, rand_iou, rand_pg]]
model_results = [
    ["AlexNet", 
     [0.74, 8.43, 0.002, 16.67, 0.006], 
     [0.60, 4.31, 0.416, 5.56, 0.080],
     [0.50, 3.06, 0.00]],
    
    ["InceptionNet-V1", 
     [0.74, 12.77, 0.000, 27.78, 0.000], 
     [0.84, 13.84, 0.000, 33.33, 0.000],
     [0.50, 2.21, 0.00]],
    
    ["ResNet-50", 
     [0.74, 6.21, 0.078, 36.11, 0.001], 
     [0.74, 5.84, 0.192, 36.11, 0.001],
     [0.50, 2.73, 5.56]],
    
    ["VGG-16", 
     [0.74, 17.55, 0.000, 13.89, 0.011], 
     [0.74, 20.78, 0.000, 11.11, 0.021],
     [0.50, 2.36, 0.00]],
]

# Find max values for bolding - these are pre-determined based on the table
max_iou = 20.78  # VGG-16 CAM
max_pg = 36.11   # ResNet-50 (both CAM and GradCAM)

# Generate the LaTeX table
table = r"""
\begin{table*}[t]
    \centering
    \setlength{\tabcolsep}{2pt}
    \caption{\textbf{Model-Expert Agreement.} Entries show IoU and PG between model-generated saliency maps and expert annotations, along with p-values and binarization thresholds (thr).}
    \begin{tabularx}{\linewidth}{l|cYY|cYY|cYY}
        \toprule
        \multirow{2}{*}{Model} & \multicolumn{3}{c|}{GradCAM} & \multicolumn{3}{c|}{CAM} & \multicolumn{3}{c}{Random} \\
        & \,thr & IoU & PG & \,thr & IoU & PG & \,thr & IoU & PG \\
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
    row = f"        {model_name} & {{\\scriptsize {gradcam_data[0]:.2f}}} & {gradcam_iou} {gradcam_iou_pval} & {gradcam_pg} {gradcam_pg_pval} & {{\\scriptsize {cam_data[0]:.2f}}} & {cam_iou} {cam_iou_pval} & {cam_pg} {cam_pg_pval} & {{\\scriptsize {random_data[0]:.2f}}} & {random_data[1]} {{\\scriptsize -}} & {random_data[2]} {{\\scriptsize -}} \\\\"
    table += row + "\n"

# Complete the table
table += r"""        \bottomrule
    \end{tabularx}
    \label{tb:ModelSaliencyComparison}
\end{table*}
"""

# Write to file
with open(OUTPUT_PATH, 'w') as f:
    f.write(table)

print(f"LaTeX table generated and saved to {OUTPUT_PATH}")

# Print the table to console for review
print("\nGenerated LaTeX Table:")
print(table)
