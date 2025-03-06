# OracleNet: Oracle Bone Script Recognition

This project implements the model proposed in the paper "OracleNet: Enhancing Oracle Bone Script Recognition with Adaptive Deformation and Texture-Structure Decoupling". OracleNet improves the accuracy of oracle bone script character recognition through adaptive deformation, texture-structure decoupling, and multi-level structured perceptual attention mechanisms.

## Features

- **Adaptive Deformation Module (ADM)**
  - Content-based adaptive control points
  - Precise local deformation control
  - Semantic integrity preservation

- **Texture-Structure Decoupling Module (TSDM)**
  - Separation of texture and structural features
  - Enhanced recognition accuracy
  - Improved handling of degraded images

- **Multi-level Structured Perceptual Attention Module (MLSPAM)**
  - Feature extraction at macro and micro levels
  - Self-attention mechanism
  - Enhanced discrimination of similar characters

## Project Structure

```
oraclenet/
├── configs/                 # Configuration files
│   └── default.yaml        # Default configuration
│
├── src/
│   ├── models/            # Model implementation
│   │   ├── adm.py        # Adaptive Deformation Module
│   │   ├── tsdm.py       # Texture-Structure Decoupling Module
│   │   ├── mlspam.py     # Multi-level Attention Module
│   │   └── oraclenet.py  # Complete OracleNet model
│   │
│   ├── data/             # Data processing
│   │   ├── dataset.py    # Dataset implementation
│   │   └── processor.py  # Data processing utilities
│   │
│   ├── loss/             # Loss functions
│   │   ├── losses.py     # Loss implementations
│   │   └── metrics.py    # Evaluation metrics
│   │
│   ├── engine/           # Training & evaluation
│   │   ├── trainer.py    # Training logic
│   │   └── evaluator.py  # Evaluation logic
│   │
│   ├── visualization/    # Visualization tools
│   │   └── visualizer.py # Visualization utilities
│   │
│   └── utils/            # Utility functions
│       ├── config.py     # Configuration
│       └── utils.py      # General utilities
│
├── tools/                # Training & evaluation scripts
│   ├── train.py         # Training script
│   └── eval.py          # Evaluation script
```

## Requirements

```bash
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.2
Pillow>=8.0.0
scipy>=1.6.0
matplotlib>=3.3.2
pyyaml>=5.4.1
scikit-learn>=0.24.1
tqdm>=4.50.2
seaborn>=0.11.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/cooljeremy/OracleNet.git
cd oraclenet

# Install dependencies
pip install -r requirements.txt
```

## Training

1. **Data Preparation**

```bash
# Create data directories
mkdir -p data/source/{train,test}
mkdir -p data/target/{train,test}

# Place your data in corresponding directories
# source/train: training data from source domain
# source/test: test data from source domain
# target/train: training data from target domain
# target/test: test data from target domain
```

2. **Configuration**

Edit `configs/default.yaml` to customize training parameters:

```yaml
# Model configuration
model:
  num_classes: 241
  hidden_dim: 256
  num_control_points: 19

# Training configuration
training:
  batch_size: 64
  num_epochs: 90
  learning_rate: 0.001
  weight_decay: 1e-4
```

3. **Start Training**

```bash
# Start training from scratch
python tools/train.py --config configs/default.yaml

# Resume training from checkpoint
python tools/train.py --config configs/default.yaml --resume checkpoints/latest_checkpoint.pth
```

## Evaluation

```bash
# Evaluate model
python tools/eval.py --config configs/default.yaml --checkpoint checkpoints/best_checkpoint.pth
```


## Visualization

The project includes various visualization tools:

- Training curves (loss and accuracy)
- Feature distribution visualization
- Attention maps visualization
- Confusion matrix

Example usage:

```python
from src.visualization.visualizer import Visualizer

visualizer = Visualizer()

# Plot training curves
visualizer.plot_training_curves(trainer.train_records)

# Visualize attention maps
visualizer.visualize_attention(image, attention_map)
```



