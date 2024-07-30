# CLIP Embedding Model Testing

This repository contains tools and scripts to test and evaluate the OpenAI CLIP (Contrastive Language-Image Pre-Training) embedding model. CLIP is a powerful model that can understand images and text in a unified way, enabling various tasks such as image classification, zero-shot learning, and image-text similarity calculations.

## Features

- **Model Loading**: Easily load the pre-trained CLIP model and its preprocessing pipeline.
- **Image Preprocessing**: Preprocess images to make them compatible with the CLIP model.
- **Text Tokenization**: Tokenize text queries for CLIP embedding.
- **Embedding Calculation**: Calculate and normalize image and text embeddings.
- **Similarity Scoring**: Compute similarity scores between image and text embeddings.
- **Visualization**: Visualize images and embeddings for better understanding and analysis.
- **CPU and GPU Support**: Run the model on both CPU and GPU seamlessly.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/clip-embedding-model-testing.git
   cd clip-embedding-model-testing

2. **Download the weights from kaggle in the relevant folder**
   [kaggle OpenAI weights](https://www.kaggle.com/datasets/titericz/openaiclipweights/code)

3. **Run the bash commands (Data paths might be different)**
   ```bash
   cp ./openaiclipweights/CLIP-main/CLIP-main/clip/bpe_simple_vocab_16e6.txt ../.venv/Lib/site-packages/clip/.
   gzip -k ../.venv/Lib/site-packages/clip/bpe_simple_vocab_16e6.txt
   ls ../.venv/Lib/site-packages/clip/bpe_simple_vocab_16e6.txt
