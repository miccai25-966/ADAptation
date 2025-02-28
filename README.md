# ADAptation
Official code for Paper ID: 966: "ADAptation: Reconstruction-based Unsupervised Active Learning for Breast Ultrasound Diagnosis".

## 📝 Description
The complete code will be released upon acceptance.

## Datasets

We use the [BUSI](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset), [BUS-BRA](https://zenodo.org/records/8231412), [UDIAT](https://www.nature.com/articles/s41597-025-04562-3), and our multi-center Dataset(MC-BUS) in our experiments, which are available below:

- BUSI: https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset
- BUS-BRA: https://zenodo.org/records/8231412
- UDIAT: https://www.nature.com/articles/s41597-025-04562-3

# How to Run the Code 🛠
### Environment Installation
### 1. Stage 1 & 2: Training ControlNet+Diffusion Model on source domain dataset and Inference.
For setting up the environment and training the source model, please refer to the [[ControlNet]]([https://github.com/whq-xxh/SFADA-GTV-Seg](https://github.com/lllyasviel/ControlNet)) project. Please note that some hyperparameters, such as the image input resolution and Condition Detector, may need to be adjusted.

### 2. Stage 3: HyperSphere-based for Contrastive Learning.

