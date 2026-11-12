# Python Programs for Machine Learning

This repository contains three beginner-level Python programs based on the
"Python programs for ML" practice material.

## Programs

### 1. K-Means Clustering on Iris
- Loads the Iris dataset from scikit-learn.
- Uses the first two features: sepal length and sepal width.
- Applies K-Means clustering with 3 clusters.
- Displays the clusters and their centroids in a scatter plot.

### 2. K-Fold Cross-Validation on Iris
- Loads the Iris dataset.
- Uses Setosa and Versicolor for binary classification.
- Trains/evaluates Logistic Regression using 10-fold cross-validation.
- Prints each fold's accuracy and the average accuracy.

### 3. Neural Network for XOR
- Creates a feedforward neural network using TensorFlow/Keras.
- Uses 2 input features, a hidden layer with 4 ReLU neurons, and 1 sigmoid output.
- Trains on the XOR sample dataset for 500 epochs.
- Prints the final loss, accuracy, and predictions, and plots training loss.

## Requirements

- Python 3.9 or newer
- See `requirements.txt` for the Python packages.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Python-Programs-for-ML.git
cd Python-Programs-for-ML
```

Create and activate a virtual environment (recommended).

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the programs

Run these commands from the repository's root directory:

```bash
python 01_KMeans_Clustering/kmeans_iris.py
python 02_KFold_Cross_Validation/kfold_iris.py
python 03_Neural_Network_XOR/xor_neural_network.py
```

## Notes

- The Iris dataset is loaded directly from scikit-learn; no separate dataset
  download is required.
- The XOR neural network is trained on a very small sample dataset. Its
  accuracy can vary between runs because the network uses random initialization.
- The extracted code has been formatted into separate runnable Python files.
  The split line in the PDF's class-name print statement was joined so the
  Python file runs correctly.

## Source

Code extracted from the uploaded PDF: **Python programs for ML.pdf**.
