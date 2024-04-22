---
license: mit
language:
 - en
metrics:
 - accuracy
pipeline_tag: image-classification
---

# Model Card for Ad Recognition Model

## Model Details

### Model Description

- **Developed by:** Kamesh Rsk (KameshRsk)
- **Model type:** Vision Transformer (ViT) for image classification
- **Language(s) (NLP):** N/A
- **License:** MIT
- **Finetuned from model:** google/vit-base-patch16-224

### Model Sources

- **Repository:** https://huggingface.co/KameshRsk/ad_recognition

## Uses

### Direct Use

This model is intended to classify images as either containing only text or containing illustrations along with text. It can be used to analyze and categorize advertisement images based on their content.

### Out-of-Scope Use

This model is trained specifically on the Illustrated Ads (Grayscale) dataset and may not perform well on other types of images or tasks.

## Bias, Risks, and Limitations

The model's performance and biases heavily depend on the training data (Illustrated Ads dataset). It may exhibit biases or limitations based on the diversity and representativeness of the dataset.

### Recommendations

Users should be aware of the potential biases and limitations of the model, especially when applying it to data different from the training distribution. Further evaluation and testing on diverse datasets is recommended.

## How to Get Started with the Model

To use this model, you can load the saved checkpoint from the Hugging Face Hub repository and make predictions on new images using the ViT model and the provided preprocessing steps.

## Training Details

### Training Data

The model was trained on the `biglam/illustrated_ads` dataset from the HuggingFace Datasets library, which contains images(Grayscale) of advertisements from various publications, along with labels indicating whether the image contains only text or illustrations.

### Training Procedure

The training procedure involves loading the dataset, preprocessing the images, splitting the data into train and test sets, and training the ViT model using PyTorch and the Accelerate library. The training process is logged to the Hugging Face Hub, where the model checkpoints are also uploaded.

#### Training Hyperparameters

- **Training regime:** Mixed precision training (fp16)
- **Optimizer:** AdamW
- **Learning rate:** 1e-5
- **Epochs:** 5

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on a held-out test set from the `biglam/illustrated_ads` dataset.

#### Metrics

The model's performance was evaluated using the accuracy metric.

### Results

#### Summary

The model achieved an accuracy of 90% on the test set after 5 epochs of training.

## Environmental Impact

- **Hardware Type:** GPU
- **Cloud Provider:** Kaggle
- **Compute Region:** N/A
- **Carbon Emitted:** Estimated to be around 0.2 kg CO2eq

## Technical Specifications

### Model Architecture and Objective

The model is a Vision Transformer (ViT) architecture adapted for image classification. The objective is to classify input images as either containing only text or containing illustrations along with text.

### Compute Infrastructure

#### Hardware

The model was trained on a NVIDIA Tesla P100 GPU provided by Kaggle.

#### Software

The model was developed using Python, PyTorch, and the Hugging Face Transformers library. The Accelerate library was used for mixed precision training and model parallelization.

## Citation

This model was developed as part of a personal project and does not have an associated paper or blog post.

## Model Card Authors

This Model Card was created by Kamesh Rsk (KameshRsk).