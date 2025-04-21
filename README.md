# Edge-Optimized Generative Image Model

## 🚀 Overview

This project explores the development of a **lightweight Generative Adversarial Network (GAN)** for synthesizing human activity images, specifically optimized for **edge devices** using **PyTorch**, **model pruning**, and **quantization**.

It also explores integration with **LLaMA Large Language Models (LLMs)** to enhance multi-modal generation for **novel, edge-compatible GenAI research**. The goal is to combine image generation and language-driven prompts to enable intelligent on-device generative capabilities.

---

## 🎯 Objectives

- 🧠 **Design a compact GAN** architecture using PyTorch for image generation tasks.
- 🔧 **Optimize the model for edge deployment** using:
  - Weight pruning (structured/unstructured)
  - Post-training quantization (INT8)
- 🤝 **Integrate LLaMA** to condition image generation with textual prompts or metadata.
- 🔄 **Investigate multi-modal GAN-LLM fusion** to synthesize data beyond standard image domains.

---

## 🧱 Project Structure

```
.
├── models/
│   ├── gan.py                  # GAN model (generator + discriminator)
│   └── llama_gan_fusion.py     # LLaMA + GAN hybrid prototype
├── training/
│   ├── train_gan.py            # Training loop for GAN
│   ├── quantize_prune.py       # Pruning and quantization scripts
├── data/
│   └── activity_dataset/       # Dataset for human activity images
├── utils/
│   └── logger.py               # Logging and visualization helpers
├── README.md
└── requirements.txt
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch >= 2.0
- torchvision
- matplotlib
- numpy
- transformers (for LLaMA)
- accelerate
- torchsummary

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Architecture

### 1. Generator

- Input: Noise vector (latent space)
- Transposed Convolution layers (Deconv)
- Batch Normalization + ReLU
- Output: 3-channel synthetic image (e.g., 64×64 RGB)

### 2. Discriminator

- Input: Real or generated image
- Convolution layers + LeakyReLU
- Fully connected + Sigmoid output (real/fake classification)

### 3. LLaMA-GAN Fusion (Optional)
- LLaMA generates a semantic embedding from a text prompt.
- GAN conditions the generation process using this embedding (via concatenation or attention-based fusion).

---

## ⚙️ Optimization for Edge Devices

- **Pruning**: Remove unimportant connections in the model using PyTorch's pruning API.

```python
from torch.nn.utils import prune
prune.l1_unstructured(layer, name='weight', amount=0.3)
```

- **Quantization**: Reduce precision from float32 to int8 to decrease memory and improve inference time.

```python
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)
```

- Evaluate optimized models using latency and model size benchmarks.

---

## 🧪 How to Run

### Training the GAN:

```bash
python training/train_gan.py
```

### Pruning and Quantization:

```bash
python training/quantize_prune.py
```

### Generate Images (Inference):

```python
from models.gan import Generator
model = Generator()
model.load_state_dict(torch.load("generator.pth"))
generate_image(model)
```

---

## 🔬 Experimental Exploration

- **LLaMA Integration**:
  - Generate semantic prompts like “a person jumping” or “man running fast”.
  - Feed these into LLaMA to get text embeddings.
  - Fuse with noise vector in GAN to control the generation.

- **Multi-Modal Generation**:
  - Future scope includes extending to video sequences and sensor data (e.g., accelerometer + image).

---

## 📌 Future Work

- 🧩 Integrate ControlNet or diffusion-based methods for fine-grained control.
- 📹 Extend to low-resolution video generation.
- 📲 Deploy final model to Raspberry Pi or Jetson Nano for edge inference tests.
- 🧪 Benchmark energy usage and frame generation latency.

---

## 📄 License

This project is licensed under the MIT License.
