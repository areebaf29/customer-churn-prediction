# Customer Churn Prediction

Predicts telecom customer churn. Logistic Regression, Random Forest and
Gradient Boosting are compared on a validation set. The selected model is
refit on train + validation and evaluated on the held-out test set.

## Run in VS Code

The saved model file is not stored in this repository. Train it once with
`train_model.py` (below), then start the app.

1. Download or clone the repository, and add the dataset (see Dataset below).
2. In VS Code choose File > Open Folder and open the project folder,
   containing `app.py` and `requirements.txt`.
3. Open Terminal > New Terminal.

Use Python 3.10 for the verified setup (tested with Python 3.10.12).
Use a clean environment with the pinned requirements.

### Windows PowerShell

If the Python launcher is available:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No PowerShell activation or execution-policy change is needed. If `py` is
not recognized, install Python 3.10 or use `python -m venv .venv` only after
`python --version` confirms Python 3.10.

### macOS / Linux

With Python 3.10 installed:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_model.py
.venv/bin/python -m streamlit run app.py
```

Open http://localhost:8501 if the browser does not open automatically.
Keep the terminal running. Press Ctrl+C in the terminal to stop the app.
Enter customer details and change the decision threshold to see how the
risk flag changes. This is a web app, not a Jupyter notebook.

`requirements.txt` pins the main scientific and app packages.
`requirements-lock.txt` is the full dependency snapshot of the verified
Python 3.10.12 environment. To recreate that exact package set, install
`-r requirements-lock.txt` instead, in a fresh Python 3.10 environment.
Verification on other operating systems was not performed.

## Train and evaluate

The commands above already run training once. To run it again, use the same
project folder and virtual environment:

Windows:
```powershell
.\.venv\Scripts\python.exe train_model.py
```

macOS / Linux:
```bash
.venv/bin/python train_model.py
```

This creates the model, metrics and plots in `outputs/`.
The script reads `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` and creates
`outputs/` if needed. Both scripts resolve resource paths relative to
where the script is stored, rather than the terminal's working folder.

## Dataset

IBM Telco Customer Churn, not stored in this repository. Download
`WA_Fn-UseC_-Telco-Customer-Churn.csv` (for example from
https://www.kaggle.com/datasets/blastchar/telco-customer-churn, which hosts
the IBM sample data) and save it as
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, creating the `data` folder.
Origin: https://github.com/IBM/telco-customer-churn-on-icp4d (see DATA_NOTICE.md)

7,043 customers and 21 columns. Cleaning removes the 11 rows with blank
TotalCharges and drops customerID: 7,032 rows, 19 predictors and the
binary Churn target (26.6% churned).

## Methodology

- Stratified 60/20/20 train/validation/test split, random_state=42.
- Scaling and one-hot encoding are fitted inside the training pipeline.
- Model selection and threshold tuning use the validation set only.
- The winner is refit on train + validation and evaluated on the held-out
  test set at the default and validation-selected thresholds.
- Test results include accuracy, precision, recall, F1, ROC-AUC and a
  confusion matrix. Permutation importance is descriptive, not causal.

## Project structure

```text
app.py
train_model.py
requirements.txt
requirements-lock.txt
README.md
DATA_NOTICE.md
data/       (add the CSV here)
outputs/    (created by train_model.py)
```

## Troubleshooting

- `No module named streamlit`: install requirements using the same
  virtual-environment Python used to launch the app.
- Model version warning or load error: create a fresh environment, install
  the pinned requirements and run train_model.py again.
- Missing data file: add the CSV at the path shown in the Dataset section.
- Missing model: run train_model.py from the project folder. Run train_model.py to create the model.
- Port 8501 busy: append `--server.port 8502` to the app command and open
  http://localhost:8502.
