# CNN-Based Construction Safety Helmet Detection

A benchmarked, explainable CNN classification system for construction 
safety helmet detection using MobileNetV2, ResNet50, and a custom CNN.

## Dataset
Safety Helmet Detection (SHWD) — Kaggle  
https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection

## Models
- Custom CNN (from scratch)
- MobileNetV2 (transfer learning)
- ResNet50 (transfer learning)

## Results
| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Custom CNN | 88.45% | 61.13% | 0.9629 |
| MobileNetV2 | 97.38% | 93.77% | 0.9960 |
| ResNet50 | 97.81% | 94.50% | 0.9959 |

## How to Run
Open `helmet_detection__1_.ipynb` in Google Colab with T4 GPU and run all cells.
