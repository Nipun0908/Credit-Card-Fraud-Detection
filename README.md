# Credit-Card-Fraud-Detection
An end-to-end Python implementation utilizing Machine Learning algorithms to detect fraudulent credit card transactions. This framework processes heavily skewed data and compares a **Logistic Regression** baseline against an ensemble **Random Forest Classifier**.
## Project Context
The dataset used contains transactions made by credit cards in September 2013 by European cardholders. 
* It presents transactions that occurred over two days, containing **284,807 transactions**.
* The dataset is highly unbalanced, with the positive class (frauds) accounting for only **0.172%** of all transactions.
* Features $V_1, V_2, \dots V_{28}$ are numerical features obtained via a Principal Component Analysis (PCA) transformation due to confidentiality constraints.
## Requirement
* numpy
* pandas 
* matplotlib.pyplot
* seaborn
## Setup Instructions
1.	Open the Repositry:
2.	Download the Dataset:
○	Download creditcard.csv from repositry.
○	Unzip and save the creditcard.csv file directly into the root directory of this repository.
3.	Install Requirements
4.	Execute Pipeline:
python code.py
## Performance Benchmark
Given the immense class imbalance, Macro Average Recall and F1-Score are prioritized over basic accuracy metrics to ensure fraudulent vectors are flagged cleanly without generating excessive false negatives.
<img width="804" height="93" alt="Screenshot (390)" src="https://github.com/user-attachments/assets/175c7e8e-2ed7-48e5-837e-ed78e8337042" />

