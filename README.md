# Text-to-Image CGAN Generator

A machine learning project that uses Conditional Generative Adversarial Networks (CGANs) to generate images from textual descriptions. This project leverages BERT embeddings for text encoding and deep learning models for image generation.

## Overview

This project implements a text-to-image generation pipeline that:

- Processes natural language descriptions using BERT (Bidirectional Encoder Representations from Transformers)
- Generates realistic images conditioned on text descriptions
- Uses the CUB-200-2011 bird dataset for training and evaluation
- Employs Conditional GANs for adversarial training

## Features

- **Text Encoding**: Uses BERT-base-uncased model to convert descriptions into semantic embeddings
- **Image Preprocessing**: Loads and resizes images to consistent dimensions (64x64 pixels)
- **Conditional Generation**: Generates images conditioned on text embeddings
- **Dataset Integration**: Works with the CUB-200-2011 dataset with comprehensive metadata including:
  - Image classifications
  - Text descriptions
  - Bounding box information
  - Training/testing splits

## Requirements

The project requires the following Python libraries:

- `torch` - Deep learning framework
- `transformers` - For BERT model and tokenizer
- `tensorflow` (optional) - For data pipeline construction
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `PIL` - Image processing

Install dependencies with:

```bash
pip install torch transformers tensorflow pandas numpy pillow
```

## Project Structure

### Main Components

1. **Dataset Setup** - Loads and merges metadata from the CUB-200-2011 dataset
2. **Data Preprocessing** - Image normalization and BERT text encoding
3. **Image Processing** - Image loading, resizing, and normalization
4. **Text Embedding** - BERT-based text-to-embedding conversion
5. **Dataset Creation** - Combining images and embeddings into batched tensors

### Key Functions

- `preprocess_images()` - Loads and preprocesses images from disk
- `generate_bert_embeddings()` - Converts text descriptions to BERT embeddings
- `create_dataset()` - Creates a TensorFlow dataset with images and embeddings

## Usage

The notebook is structured in sections:

1. **Dataset Setup** - Initialize and load metadata
2. **Data Preprocessing** - Prepare images and text encodings
3. **BERT Embeddings** - Generate semantic embeddings for text
4. **Image Processing** - Preprocess images to consistent dimensions
5. **Dataset Creation** - Create batched dataset for training
6. **Model Training** - Train the CGAN model
7. **Image Generation** - Generate new images from text descriptions
8. **Evaluation** - Evaluate generation quality

## Dataset

This project uses the **CUB-200-2011 (Caltech-UCSD Birds 200-2011)** dataset:

- 11,788 bird images
- 200 bird species
- Detailed annotations including bounding boxes
- Train/test split included

Expected file structure:

```text
CUB_200_2011/
├── classes.txt
├── images.txt
├── train_test_split.txt
├── image_class_labels.txt
├── bounding_boxes.txt
└── images/
    └── [bird species]/
        └── [image files]
```

## Model Architecture

### BERT Text Encoder

- Model: `bert-base-uncased`
- Output: 768-dimensional embeddings
- Uses CLS token embedding for sentence-level representation

### CGAN Components

- **Generator**: Generates 64x64 RGB images conditioned on text embeddings
- **Discriminator**: Evaluates authenticity of generated images given text conditioning

## Training

The model uses adversarial training with the following features:

- Batch processing with configurable batch size (default: 32)
- Data shuffling and prefetching for efficiency
- Conditional generation based on text embeddings

## Output

The model generates realistic bird images based on text descriptions, such as:

- "The Laysan Albatross is a large seabird with white feathers and black wings"
- "This red bird has a distinctive crest and sings loudly in the morning"

## Notes

- Images are resized to 64x64 pixels and normalized to [0, 1] range
- Text descriptions are padded/truncated to 128 tokens
- The project uses GPU acceleration when available (CUDA)
- Kaggle dataset integration for streamlined data access

## Future Improvements

- Increase image resolution to 128x128 or higher
- Implement additional evaluation metrics (FID, Inception Score)
- Add attention mechanisms for better text-image alignment
- Fine-tune BERT for domain-specific bird descriptions
- Implement multi-GPU training for faster convergence

## References

- BERT: [Attention Is All You Need](https://arxiv.org/abs/1810.04805)
- Conditional GANs: [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
- CUB-200-2011 Dataset: [The Caltech-UCSD Birds-200-2011 Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

## License

This project is provided for educational purposes.
