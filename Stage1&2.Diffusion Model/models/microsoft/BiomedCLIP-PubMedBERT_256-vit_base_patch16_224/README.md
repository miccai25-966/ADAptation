---
language: en
tags:
- clip
- biology
- medical
license: mit
library_name: open_clip
widget:
  - src: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/squamous_cell_carcinoma_histopathology.jpeg
    candidate_labels: adenocarcinoma histopathology, squamous cell carcinoma histopathology
    example_title: squamous cell carcinoma histopathology
  - src: >-
      https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/adenocarcinoma_histopathology.jpg
    candidate_labels: adenocarcinoma histopathology, squamous cell carcinoma histopathology
    example_title: adenocarcinoma histopathology    
  - src: >-
      https://upload.wikimedia.org/wikipedia/commons/5/57/Left-sided_Pleural_Effusion.jpg
    candidate_labels: left-sided pleural effusion chest x-ray, right-sided pleural effusion chest x-ray, normal chest x-ray
    example_title: left-sided pleural effusion chest x-ray
pipeline_tag: zero-shot-image-classification
---

# BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

[BiomedCLIP](https://aka.ms/biomedclip-paper) is a biomedical vision-language foundation model that is pretrained on [PMC-15M](https://aka.ms/biomedclip-paper), a dataset of 15 million figure-caption pairs extracted from biomedical research articles in PubMed Central, using contrastive learning. 
It uses PubMedBERT as the text encoder and Vision Transformer as the image encoder, with domain-specific adaptations.
It can perform various vision-language processing (VLP) tasks such as cross-modal retrieval, image classification, and visual question answering. 
BiomedCLIP establishes new state of the art in a wide range of standard datasets, and substantially outperforms prior VLP approaches:

![](biomed-vlp-eval.svg)


## Citation

```bibtex
@misc{https://doi.org/10.48550/arXiv.2303.00915,
  doi = {10.48550/ARXIV.2303.00915},
  url = {https://arxiv.org/abs/2303.00915},
  author = {Zhang, Sheng and Xu, Yanbo and Usuyama, Naoto and Bagga, Jaspreet and Tinn, Robert and Preston, Sam and Rao, Rajesh and Wei, Mu and Valluri, Naveen and Wong, Cliff and Lungren, Matthew and Naumann, Tristan and Poon, Hoifung},
  title = {Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing},
  publisher = {arXiv},
  year = {2023},
}
```

## Model Use

### 1. Environment

```bash
conda create -n biomedclip python=3.10 -y
conda activate biomedclip
pip install open_clip_torch==2.23.0 transformers==4.35.2 matplotlib
```

### 2.1 Load from HF hub

```python
import torch
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# Load the model and config files from the Hugging Face Hub
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


# Zero-shot image classification
template = 'this is a photo of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]

dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
test_imgs = [
    'squamous_cell_carcinoma_histopathology.jpeg',
    'H_and_E_histopathology.jpg',
    'bone_X-ray.jpg',
    'adenocarcinoma_histopathology.jpg',
    'covid_line_chart.png',
    'IHC_histopathology.jpg',
    'chest_X-ray.jpg',
    'brain_MRI.jpg',
    'pie_chart.png'
]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

context_length = 256

images = torch.stack([preprocess(Image.open(urlopen(dataset_url + img))) for img in test_imgs]).to(device)
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
with torch.no_grad():
    image_features, text_features, logit_scale = model(images, texts)

    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

top_k = -1

for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]

    top_k = len(labels) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{labels[jth_index]}: {logits[i][jth_index]}')
    print('\n')
```

### 2.2 Load from local files

```python
import json

from urllib.request import urlopen
from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS


# Download the model and config files
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_pytorch_model.bin",
    local_dir="checkpoints"
)
hf_hub_download(
    repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    filename="open_clip_config.json",
    local_dir="checkpoints"
)


# Load the model and config files
model_name = "biomedclip_local"

with open("checkpoints/open_clip_config.json", "r") as f:
    config = json.load(f)
    model_cfg = config["model_cfg"]
    preprocess_cfg = config["preprocess_cfg"]


if (not model_name.startswith(HF_HUB_PREFIX)
    and model_name not in _MODEL_CONFIGS
    and config is not None):
    _MODEL_CONFIGS[model_name] = model_cfg

tokenizer = get_tokenizer(model_name)

model, _, preprocess = create_model_and_transforms(
    model_name=model_name,
    pretrained="checkpoints/open_clip_pytorch_model.bin",
    **{f"image_{k}": v for k, v in preprocess_cfg.items()},
)


# Zero-shot image classification
template = 'this is a photo of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]

dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
test_imgs = [
    'squamous_cell_carcinoma_histopathology.jpeg',
    'H_and_E_histopathology.jpg',
    'bone_X-ray.jpg',
    'adenocarcinoma_histopathology.jpg',
    'covid_line_chart.png',
    'IHC_histopathology.jpg',
    'chest_X-ray.jpg',
    'brain_MRI.jpg',
    'pie_chart.png'
]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

context_length = 256

images = torch.stack([preprocess(Image.open(urlopen(dataset_url + img))) for img in test_imgs]).to(device)
texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)
with torch.no_grad():
    image_features, text_features, logit_scale = model(images, texts)

    logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()
    sorted_indices = sorted_indices.cpu().numpy()

top_k = -1

for i, img in enumerate(test_imgs):
    pred = labels[sorted_indices[i][0]]

    top_k = len(labels) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{labels[jth_index]}: {logits[i][jth_index]}')
    print('\n')

```

### Use in Jupyter Notebook

Please refer to this [example notebook](https://aka.ms/biomedclip-example-notebook).

### Intended Use

This model is intended to be used solely for (I) future research on visual-language processing and (II) reproducibility of the experimental results reported in the reference paper.

#### Primary Intended Use

The primary intended use is to support AI researchers building on top of this work. BiomedCLIP and its associated models should be helpful for exploring various biomedical VLP research questions, especially in the radiology domain.

#### Out-of-Scope Use

**Any** deployed use case of the model --- commercial or otherwise --- is currently out of scope. Although we evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are not intended for deployed use cases. Please refer to [the associated paper](https://aka.ms/biomedclip-paper) for more details.

## Data

This model builds upon [PMC-15M dataset](https://aka.ms/biomedclip-paper), which is a large-scale parallel image-text dataset for biomedical vision-language processing. It contains 15 million figure-caption pairs extracted from biomedical research articles in PubMed Central. It covers a diverse range of biomedical image types, such as microscopy, radiography, histology, and more.

## Limitations

This model was developed using English corpora, and thus can be considered English-only.

## Further information

Please refer to the corresponding paper, ["Large-Scale Domain-Specific Pretraining for Biomedical Vision-Language Processing"](https://aka.ms/biomedclip-paper) for additional details on the model training and evaluation.