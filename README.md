# Do AI and Human Experts See Pneumonia the Same Way?

This repository contains the code and results for a research project investigating the explainability and human alignment of deep learning models for pneumonia detection in chest X-rays.

*We would like to thank our medical annotators:
Carmen, Claudia, Diana, Francesca, Giovanni P., Lorenzo, Luca G., Matilde M., Sara, Simone P., Tommaso D.. Their medical expertise and generous support in annotating the images were essential for the success of this project.*

## Overview

Deep learning has achieved remarkable performance in medical image analysis, but the opacity of neural network decision-making remains a critical barrier to clinical adoption. This project examines the explainability of convolutional neural networks (CNNs) in diagnosing pneumonia from pediatric chest X-ray images by comparing the regions they deem important with those identified by medical experts.

1. **How do different CNN architectures compare in pneumonia classification performance?**
2. **To what extent do model-generated saliency maps align with regions identified by medical experts?**

### Methodology

We trained and evaluated four CNN architectures:
   - AlexNet (baseline)
   - VGG-16 (deep architecture with uniform 3×3 convolutions)
   - InceptionNet-V1 (network with inception modules for multi-scale feature extraction) 
   - ResNet-50 (deep residual network with skip connections)

Each model was trained both from scratch and with ImageNet pretraining.

Medical professionals annotated 50 pneumonia-positive chest X-rays, marking regions they considered diagnostically relevant. We measured inter-expert agreement using IoU and created consensus maps through annotation intersection.

We generated visual explanations using Class Activation Mapping (CAM) and Gradient-weighted Class Activation Mapping (Grad-CAM)

We compared model-generated saliency maps with expert consensus using:
   - Intersection over Union (IoU)
   - Pointing Game (checking if the model's focus point falls within expert-annotated regions)

### Key Findings
The best-performing model is not the most aligned with expert annotations, and some very good predictors are very bad in alignment metrics. This suggests that high classification performance does not necessarily imply high interpretability or alignment with human reasoning, highlighting the need to evaluate both accuracy and explainability when assessing medical AI systems.

## Project Setup and Usage

### 1. Installation

Install the necessary Python packages using `pip`:

```bash
pip install -r requirements.txt
```

This project requires Python 3.8+ and PyTorch 1.8+.

### 2. Data Preparation

Download the pediatric chest X-ray dataset used in this study from [Kermany et al., 2018](https://data.mendeley.com/datasets/rscbjbr9sj/3).

You can use the provided script:

```bash
python medimg-saliency-benchmark/data/download.py
```

The data will be automatically organized into train, validation, and test directories.

### 3. Training Models

Configure your training by editing `config.json`:

```json
{
    "model": "an",          # Model architecture code: 
                            # an = AlexNet, vgg = VGG-16, 
                            # in = InceptionNet-V1, rn = ResNet-50
    "pretrained": true,     # Whether to use ImageNet pretrained weights
    "linear": true,         # Whether to replace classifier with GAP + Linear layer
                            # (required for CAM, already present in ResNet-50 and InceptionNet-V1)
    "batch_size": 32,       # Training/validation/test batch size
    "epochs": 10,           # Maximum training epochs
    "max_lr": 0.001,        # Maximum learning rate (OneCycle scheduler)
    "pct_start": 0.1,       # Percentage of total steps for learning rate warmup
    "wandb": false          # Whether to log metrics to Weights & Biases
}
```

To train a model:

```bash
python medimg-saliency-benchmark/train.py path/to/config.json
```

### 4. Testing Models

Evaluate all trained models in the `checkpoints/` directory:

```bash
python medimg-saliency-benchmark/test.py path/to/config.json
```

### 5. Analyzing Expert Annotations

Process expert annotations and measure agreement metrics:

```bash
python medimg-saliency-benchmark/annotations.py
```

### 6. Evaluating Saliency Agreement

Calculate and analyze the agreement between model saliency maps and expert annotations:

```bash
python medimg-saliency-benchmark/agreement.py
```

### 7. Correlation Analysis

Analyze correlation between model performance and expert alignment:

```bash
python medimg-saliency-benchmark/correlation.py
```
## Repository Structure

```bash
├── LICENSE              
├── README.md           
├── requirements.txt       
└── medimg-saliency-benchmark/     
    ├── agreement.py        # Calculates alignment between saliency maps and expert annotations
    ├── annotations.py      # Processes and analyzes expert annotations
    ├── config.json         
    ├── correlation.py      # Analyzes correlation between performance and alignment
    ├── datamodule.py      
    ├── gridsearch.py       # Hyperparameter optimization
    ├── main.ipynb          # Main notebook for exploratory analysis
    ├── models.py          
    ├── saliency.py        
    ├── test.py             # Model evaluation script
    ├── threshold.py        # Threshold optimization for saliency map binarization
    ├── train.py          
    ├── utils.py            # Utility functions
    ├── checkpoints/        # Saved model weights
    ├── data/              
    │   ├── annotations/  
    │   │   ├── annotated/  
    │   │   └── original/   
    │   ├── download.py     # Dataset download script
    │   ├── test/           
    │   ├── train/          
    │   └── val/           
    ├── evaluation/        
    │   ├── *.json          # Model performance metrics
    │   ├── *.csv           # Agreement and correlation results
    │   └── *.tex           # Generated LaTeX tables
    └── plots/              # Generated visualizations and figures
```

## Results and Findings

### Model Performance

Our experiments show that pretrained models consistently outperform their non-pretrained counterparts across all architectures. ResNet-50 with pretraining achieved the best overall performance (94.87% accuracy, 96.00% F1 score).

| Model | Adapted | Pretrained | Accuracy | F1 | ROC AUC | Specificity |
|-------|---------|------------|----------|------|---------|-------------|
| AlexNet | Yes | Yes | 90.34% | 92.72% | 96.72% | 76.40% |
| VGG-16 | Yes | Yes | 91.46% | 93.50% | 97.93% | 79.85% |
| InceptionNet-V1 | - | Yes | 91.84% | 93.79% | 98.06% | 80.37% |
| ResNet-50 | - | Yes | **94.87%** | **96.00%** |**98.24%**| **88.86%**|

### Expert Alignment

In terms of explainability, we observed an interesting divergence between the IoU and Pointing Game metrics:

- VGG-16 achieved the highest IoU scores, suggesting its saliency maps align more consistently with expert annotations.
- ResNet-50 and InceptionNet-V1 performed best on the Pointing Game metric, indicating their ability to localize the most diagnostically relevant points.

This suggests that different architecture designs promote different types of feature attention patterns, with important implications for explainable AI in medical imaging.

## Expert Annotation Tool

To facilitate the collection of expert annotations, we developed a custom web-based tool (https://gciro.pythonanywhere.com) for medical professionals to identify relevant regions in chest X-ray images. The tool provided a simple drawing interface with adjustable brush sizes, allowing experts to highlight regions they considered diagnostically significant.

## References

* **Dataset:** Kermany, D.S., Goldbaum, M., Cai, W. et al. (2018). _Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning_. Cell. [https://doi.org/10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
* **CAM:** Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). _Learning Deep Features for Discriminative Localization_. CVPR. ([arXiv:1512.04150](https://arxiv.org/abs/1512.04150))
* **Grad-CAM:** Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). _Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization_. ICCV. ([arXiv:1610.02391](https://arxiv.org/abs/1610.02391))
* **CNN Architectures:**
    * **AlexNet:** Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). _ImageNet Classification with Deep Convolutional Neural Networks_. NIPS. ([DOI](https://doi.org/10.1145/3065386))
    * **VGG-16:** Simonyan, K., & Zisserman, A. (2014). _Very Deep Convolutional Networks for Large-Scale Image Recognition_. ICLR. ([arXiv:1409.1556](https://arxiv.org/abs/1409.1556))
    * **InceptionNet:** Szegedy, C., Liu, W., Jia, Y., et al. (2015). _Going Deeper with Convolutions_. CVPR. ([arXiv:1409.4842](https://arxiv.org/abs/1409.4842))
    * **ResNet:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). _Deep Residual Learning for Image Recognition_. CVPR. ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))