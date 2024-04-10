# PyTorchClassifier
## Binary Classifier with PyTorch
This script trains a neural network model based on a provided dataset. It preprocesses the data, defines the model architecture, trains the model, and evaluates its performance.

## Requirements

- Python 3.x
- pandas
- torch
- scikit-learn

## Installation

1. Clone the repository:
```
git clone https://github.com/shri30yans/PyTorchClassifier.git
```
```
cd PyTorchClassifier
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Place your dataset in CSV format in the root directory with the name `dataset.csv`.
2. Modify the configuration parameters in the `main()` function of `main.py` according to your dataset and requirements.
3. Run the script:
```
python main.py
```

The script will preprocess the data, train the model, and output the accuracy on the validation set.

