# CNN Saliency Evaluation for Medical Imaging

This project aims to evaluate and benchmark the localization accuracy of saliency methods, specifically Grad-CAM, generated from different Convolutional Neural Network (CNN) architectures. The focus is on a medical image classification task: detecting COVID-19 in chest X-rays against normal cases, using the [Chest X-Ray (Covid-19 & Pneumonia) dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia).

Standard saliency methods like Grad-CAM provide visual explanations for CNN predictions, highlighting input regions deemed important. However, their alignment with true pathological features, especially across different model backbones, requires careful evaluation. Lacking publicly available pixel-level ground truth for this specific dataset, we will generate reference annotations manually for a subset of the test data.

Our goal is to train several common CNN architectures (e.g., ResNet50, DenseNet121, VGG16, MobileNetV2) using transfer learning for the COVID-19 vs. Normal classification task. We will then generate Grad-CAM maps for correctly classified test images from our annotated subset and quantitatively evaluate how well these maps overlap with the manually identified regions of interest using metrics like Intersection over Union (IoU) and the Pointing Game hit rate. This comparison will help understand if certain architectures inherently produce more accurate saliency-based localizations for this task.

## TO-DOs
- [x] Set up initial project structure and repository.
- [ ] Finalize `config.json` structure and parameters.
- [x] Implement data download script (`scripts/download_data.py`).
- [ ] Run data download script to obtain the dataset.
- [ ] Implement annotation script (`scripts/annotate_images.py`).
- [ ] Select ~50 test images (COVID/Normal) for the annotation subset.
- [ ] Perform manual annotation using the script and consolidate results (`data/annotations.csv`).
- [ ] Implement utility function to convert annotations (clicks/boxes) to masks.
- [ ] Implement data loaders (`src/datamodule.py`) for binary classification (COVID vs. Normal) with augmentation.
- [ ] Implement model building functions (`src/models.py`) for ResNet50, DenseNet121, VGG16, MobileNetV2 using transfer learning.
- [ ] Implement training script (`src/train.py`) with fine-tuning strategy and callbacks.
- [ ] Train all specified models and save best checkpoints.
- [ ] Implement Grad-CAM generation using `tf-keras-vis` (or `captum`).
- [ ] Implement evaluation script (`src/evaluate.py`) calculating IoU and Pointing Game metrics against annotation masks.
- [ ] Run evaluation script to generate results.
- [ ] Draft the project report using the provided LaTeX template.
- [ ] Complete helper functions in `src/utils.py`.
- [ ] Refine README and add visualizations/results summary.

## Key Decisions / Next Steps
- Verify data download and structure.
- Coordinate manual annotation process among team members.
- Finalize the exact set of hyperparameters for training in `config.json`.
- Decide on the specific thresholding strategy for Grad-CAM maps for IoU calculation.

## Usage

### 1. Installation
Install the necessary Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation
First, download the dataset using the provided script. This requires having your Kaggle credentials set up correctly (`~/.kaggle/kaggle.json`).

```bash
python scripts/download_data.py
```
This will download the dataset to the path specified within the script (e.g., `data/raw/`).


## Misc
1.  Create a `local_config.json` (added to `.gitignore`) for local testing without modifying the main `config.json`.
2.  If using Weights & Biases (`wandb`), log in via the CLI:
    ```bash
    python -c "import wandb; wandb.login()"
    ```

## Repository Structure (Planned)
```bash
├── checkpoints/            
├── data/
│   ├── raw/                # (ignored by git)
│   ├── annotation_subset/  # Subset of images for manual annotation
│   └── annotations.csv                    
├── results/
│   ├── saliency_maps/      # Example saved saliency map visualizations
│   └── saliency_evaluation_summary.csv
├── scripts/
│   ├── download_data.py    
│   └── annotate_images.py  
├── src/
│   ├── __init__.py
│   ├── config.json
│   ├── datamodule.py       
│   ├── evaluate.py         
│   ├── models.py           
│   ├── train.py            
│   └── utils.py            
├── notebooks/              # Jupyter notebooks for exploration, visualization (optional)
├── .gitignore           
├── LICENSE                 
├── README.md               
└── requirements.txt        
```

## References (Preliminary)

* **Dataset:** Prashant Patel (2021). _Chest X-Ray (Covid-19 & Pneumonia)_. Kaggle. [https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia) (Acknowledges original sources within)
* **Grad-CAM:** Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). _Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization_. ICCV. ([arXiv:1610.02391](https://arxiv.org/abs/1610.02391))
* **(Models - Examples)**
    * He, K., Zhang, X., Ren, S., & Sun, J. (2016). _Deep Residual Learning for Image Recognition_. CVPR. ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
    * Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). _Densely Connected Convolutional Networks_. CVPR. ([arXiv:1608.06993](https://arxiv.org/abs/1608.06993))
* **(Benchmarking Saliency - Example)** Saporta, A., et al. (2022). _Benchmarking saliency methods for chest X-ray interpretation_. Nature Machine Intelligence. ([DOI](https://doi.org/10.1038/s42256-022-00536-x)) - *For methodology inspiration.*