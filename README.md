# Task 1 — Iris Flower Classification

**Internship:** Oasis Infobyte Data Science Internship (OIBSIP)
**Track:** Data Science

## Objective
Train a machine learning classification model to identify the species of an iris flower (Setosa, Versicolor, or Virginica) from its physical measurements.

## Dataset
The Iris dataset, loaded directly from `sklearn.datasets.load_iris()` — no external download needed. It contains 150 samples with 4 features each (sepal length, sepal width, petal length, petal width) and 3 target classes, 50 samples per class.

## Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Approach
1. Loaded the dataset and converted it into a labeled DataFrame.
2. Performed EDA — checked shape, data types, missing values, and descriptive statistics.
3. Visualized feature distributions using a pairplot and per-feature box plots grouped by species.
4. Identified petal length and petal width as the most discriminative features based on the visualizations.
5. Split the data into training and test sets (80/20, stratified).
6. Trained three classifiers: Logistic Regression, K-Nearest Neighbours, and Decision Tree.
7. Evaluated each model using accuracy, confusion matrix, and classification report (precision, recall, F1-score).

## Results

| Model | Accuracy |
|---|---|
| K-Nearest Neighbours | 100% |
| Logistic Regression | 96.7% |
| Decision Tree | 93.3% |

**Best model: K-Nearest Neighbours (KNN)** — it classified every test sample correctly across all three species, with no misclassifications even in the overlapping Versicolor/Virginica region where the other two models made small errors.

## Files
- `Shalini Naga Dhonthu Balla_Task1.ipynb` — full notebook with EDA, visualizations, model training, and evaluation.

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Open the notebook in Jupyter and run all cells.
