# BATD - Backdoor Attacks on Tabular Data

This repository contains code for implementing and evaluating backdoor attacks on tabular data models.

## Project Structure

```
BATD/
├── src/                    # Source code package
│   ├── models/             # Model implementations
│   ├── dataset/            # Dataset implementations
│   ├── defense/            # Defense mechanisms
│   │   ├── pruning.py      # Pruning-based defense
│   │   ├── ss.py           # Spectral Signature defense
│   │   ├── NC.py           # Neural Cleanse defense
│   │   └── nc_files/       # Neural Cleanse utilities
│   ├── attack.py           # Attack implementation
│   ├── feature_importance.py # Feature importance analysis
├── data/                   # Data files
├── saved_models/           # Saved model checkpoints
├── saved_triggers/         # Saved trigger patterns
├── saved_datasets/         # Saved processed datasets
├── results/                # Experimental results
└── setup.py                # Package installation script
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd BATD

# Install the package in development mode
pip install -e .
```

## Usage

```python
# Example usage
from src.models import FTTModel, TabNetModel, SAINTModel
from src.dataset import BankMarketing, ACI, HIGGS
from src.attack import Attack
from src.defense import NeuralCleanse, SpectralSignature
from src.defense.nc_files import reverse_engineer_trigger, mad_outlier_detection

# Load a dataset
dataset = BankMarketing()

# Initialize a model
model = FTTModel(dataset.get_num_features(), dataset.get_num_classes())

# Create an attack
attack = Attack(device="cuda", model=model, data_obj=dataset, target_label=1, 
                mu=0.1, beta=0.5, lambd=0.1, epsilon=0.1)

# Train the model
attack.train(dataset)

# Generate poisoned data
poisoned_dataset = attack.construct_poisoned_dataset(dataset)

# Apply defense mechanisms
nc = NeuralCleanse(model, dataset)
nc_results = nc.detect()

ss = SpectralSignature(model, dataset)
ss_results = ss.detect()
```

## License

[Add license information here]
