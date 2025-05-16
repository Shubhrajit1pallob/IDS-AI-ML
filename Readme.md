# Intrusion Detection System Using ML and RL

## Overview

This project implements an Intrusion Detection System (IDS) using a combination of Machine Learning (ML), Deep Learning (DL), and Reinforcement Learning (RL) techniques. The goal is to accurately detect and classify network intrusions using advanced AI models.

- **Course:** CSE 543 - Information Assurance and Security
- **Group:** 10

## Features

- **Data Preprocessing:** Handles missing values, label encoding, and feature selection.
- **Machine Learning Models:** Implements and evaluates multiple classifiers including SVC, Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and more.
- **Deep Learning:** Uses Keras/TensorFlow for multi-class attack classification.
- **Reinforcement Learning:** Implements a DQN-based RL agent for attack-type classification using Stable Baselines3 and PyTorch.
- **Ensemble Methods:** Includes Voting Classifier and Majority Voting Classifier (MVC) for improved accuracy.
- **Evaluation:** Provides detailed metrics such as accuracy, precision, recall, and classification reports.

## Project Structure

- [`IAS_Paper_Project_GROUP_10_Implementation_DL_ML_RL.ipynb`](IAS_Paper_Project_GROUP_10_Implementation_DL_ML_RL.ipynb): Main Jupyter notebook containing all code, experiments, and results.
- `IAS-Paper-Project_GROUP-10_DL-ML-RL_PaperPDF.pdf`: Project report and documentation.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, tensorflow, torch, stable-baselines3, gym, xgboost, lightgbm

You can install the dependencies using pip:

```sh
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow torch stable-baselines3 gym xgboost lightgbm
```

### Running the Project

1. Download the dataset as specified in the notebook (e.g., NF-UQ-NIDS-v2 from Kaggle).
2. Open [`IAS_Paper_Project_GROUP_10_Implementation_DL_ML_RL.ipynb`](IAS_Paper_Project_GROUP_10_Implementation_DL_ML_RL.ipynb) in Jupyter Notebook.
3. Run the notebook cells sequentially to preprocess data, train models, and evaluate results.

## Results

- The notebook provides detailed output for each model, including cross-validation scores and confusion matrices.
- Ensemble and RL-based approaches are compared for effectiveness in intrusion detection.

## Authors

- Abhay Jogenipalli
- Kumar Hasti
- Kruthika Suresh
- Sakshi Sheth
- Rahul Tallam
- Rushir Bhavsar
- Subharajit Pallob
- Varshil Shah

## License

This project is for academic purposes.

---

For more details, see the [project notebook](IAS_Paper_Project_GROUP_10_Implementation_DL_ML_RL.ipynb) and [project report](IAS-Paper-Project_GROUP-10_DL-ML-RL_PaperPDF.pdf).
