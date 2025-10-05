Credit Approval with Logistic Regression
========================================

Project Overview
----------------
This project develops a binary classification model to predict whether a credit application will be approved using Logistic Regression.
The model demonstrates the complete machine learning workflow — from dataset exploration to preprocessing, model training, evaluation,
and interpretation — implemented in Python using scikit-learn.

The purpose is to practice applied data science skills and build a transparent, auditable model that meets fintech requirements for responsible modeling.

----------------------------------------------------------------------
Key Objectives
----------------------------------------------------------------------
1. **Perform Exploratory Data Analysis (EDA)** on the dataset to identify variable types, distribution shapes, correlations, and missing values.
2. **Preprocess data** using ColumnTransformer and Pipeline:
   - Numeric features: imputation (median) and standardization.
   - Categorical features: imputation (most frequent) and one-hot encoding.
3. **Train a Logistic Regression model** with configurable hyperparameters and class-weight balancing.
4. **Interpret coefficients** as odds ratios and identify influential predictors.
5. **Evaluate model performance** using multiple metrics and visualization.
6. **Analyze threshold trade-offs** to guide decision-making in credit approval risk contexts.
7. **Ensure responsible modeling** by excluding or documenting sensitive attributes.

----------------------------------------------------------------------
Folder Structure
----------------------------------------------------------------------
credit-approval-logreg/
├── src/
│   └── main.py                 # Main Python script with full pipeline
├── figures/                    # Auto-generated EDA and evaluation plots
├── outputs/                    # Model metrics and coefficient exports
├── data/                       # Placeholder for input CSV dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Short Markdown version
├── README.txt                  # Detailed project documentation (this file)
└── .gitignore                  # Ignores virtual environment and outputs

----------------------------------------------------------------------
Environment Setup
----------------------------------------------------------------------
1. **Create a virtual environment**
   macOS/Linux:
       python3 -m venv .venv
       source .venv/bin/activate

   Windows (PowerShell):
       py -3 -m venv .venv
       .\.venv\Scripts\Activate.ps1

2. **Install dependencies**
       pip install -r requirements.txt

3. **Verify installation**
       python src/main.py --help

----------------------------------------------------------------------
How to Run the Model
----------------------------------------------------------------------
**Option 1: Synthetic Dataset (no data needed)**
    python src/main.py --generate-synthetic --random-state 42 --balance --target-precision 0.80

    This automatically generates a 2000-sample dataset with both numeric and categorical variables.
    The pipeline runs end-to-end and saves all artifacts to /figures and /outputs.

**Option 2: Use Your Own Dataset**
    python src/main.py --data-path data/credit.csv --target Approved --positive-class yes --balance --target-precision 0.85

    Required arguments:
        --data-path            Path to CSV or Excel file
        --target               Target column name
        --positive-class       Label that represents approval (positive outcome)

    Optional arguments:
        --drop-cols             Columns to drop (IDs, leaks)
        --sensitive-cols        Columns to exclude for fairness
        --penalty               Regularization type (l1 or l2)
        --C                     Regularization strength (default 1.0)
        --threshold             Manual decision threshold
        --target-precision      Auto-selects smallest threshold achieving desired precision

----------------------------------------------------------------------
Model Details
----------------------------------------------------------------------
- **Algorithm:** Logistic Regression (binary classification)
- **Solver:** lbfgs (L2) or liblinear (L1)
- **Regularization:** Configurable via parameter C
- **Class Weight:** 'balanced' option compensates for class imbalance
- **Preprocessing:** StandardScaler + OneHotEncoder inside ColumnTransformer
- **Outputs:**
    - coefficients_oddsratios.csv → Coefficients & odds ratios
    - evaluation.json → Model metrics and metadata
    - Plots → ROC curve, PR curve, confusion matrix, threshold trade-off

----------------------------------------------------------------------
Evaluation Metrics
----------------------------------------------------------------------
The model computes and saves the following metrics automatically:

| Metric        | Meaning                                           |
|----------------|---------------------------------------------------|
| Accuracy       | Overall fraction of correct predictions          |
| Precision      | Fraction of predicted approvals that were correct|
| Recall         | Fraction of actual approvals correctly identified|
| F1 Score       | Harmonic mean of precision and recall            |
| ROC-AUC        | Area under Receiver Operating Characteristic     |
| PR-AUC         | Area under Precision-Recall curve                |

Additional figures include:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Threshold Trade-off Curve
- Missingness Plot
- Feature Histograms
- Correlation Heatmap

----------------------------------------------------------------------
Interpretation
----------------------------------------------------------------------
Logistic regression coefficients are interpreted as changes in log-odds.
The odds ratio (exp(coefficient)) represents the multiplicative effect on approval odds for a one-unit increase in that variable.

Example:
    If "income" has an odds ratio of 1.15,
    each standardized unit increase in income multiplies approval odds by 1.15, all else equal.

----------------------------------------------------------------------
Responsible Modeling
----------------------------------------------------------------------
- The model allows exclusion of sensitive features (e.g., gender, race, age) via the `--sensitive-cols` flag.
- Categorical variables that act as proxies (e.g., ZIP code, employment length) should be reviewed carefully.
- Document assumptions and justify feature inclusion in any deployment scenario.

----------------------------------------------------------------------
Troubleshooting
----------------------------------------------------------------------
- If plots don’t appear, check write permissions in `figures/` and `outputs/`.
- If precision or recall is 0, reduce threshold (e.g., `--threshold 0.3`).
- If training time is long, reduce samples (`--n-samples 1000`).

----------------------------------------------------------------------
References (APA style)
----------------------------------------------------------------------
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
- scikit-learn developers. (2025). *LogisticRegression.* https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- IBM. (2025). *What is Logistic Regression?* https://www.ibm.com/topics/logistic-regression
- Kaggle Datasets. (2025). *Credit Approval Data.* https://www.kaggle.com/datasets/
- UCI Machine Learning Repository. (2025). *Credit Approval Dataset.* https://archive.ics.uci.edu/
