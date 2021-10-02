# Multi-modal classification of DME patients using machine learning
Diabetic retinopathy (DME) is orders of magnitude more complex than neovascular macular degeneration. It is difficult to determine what factors contribute to achievement of long-term clinical outcomes for patients. Similarly, erratic treatment protocols make it challenging to classify and correct doctor decisions. Here, we apply both supervised and unsupervised learning techniques to
1. Assess the characteristics of treatment that lead to good patient vision, in the [Baseline](https://github.com/charlieoneill11/diabeticretinopathy/blob/main/notebooks/Baseline.ipynb) notebook.
2. Determine how accurately we can predict visual changes with minimal data, in the [Prediction](https://github.com/charlieoneill11/diabeticretinopathy/blob/main/notebooks/Prediction.ipynb) notebook.
3. Group doctor protocols in a definitional way, in the [Unsupervised](https://github.com/charlieoneill11/diabeticretinopathy/blob/main/notebooks/Unsupervised.ipynb) notebook.

## Repository structure
The repository follows a standard structure:
* [input](https://github.com/charlieoneill11/diabeticretinopathy/tree/main/input) contains both the patient data as well as the notebooks used to clean and feature engineer. The main datasets for prediction are `df_one_year.csv`, which has the accompanying kfold cross-validation grouped dataset `df_one_year_fold.csv`. Train and test splits, of the same name, have also been provided.
* [notebooks](https://github.com/charlieoneill11/diabeticretinopathy/tree/main/notebooks) contains notebooks used for experimenting and producing results. The notebooks are in order of the above three tasks.
* [src](https://github.com/charlieoneill11/diabeticretinopathy/tree/main/src) contains the Python scripts allowing the user to train and evaluate different models on the required dataset from the command line. The main script is [train.py](https://github.com/charlieoneill11/diabeticretinopathy/blob/main/src/train.py), which relies on the other scripts for configuration, dataset retrieval and argument parsing.

## Results
The results of the analysis is compiled in [Report.pdf](https://github.com/charlieoneill11/diabeticretinopathy/blob/main/Report.pdf).
