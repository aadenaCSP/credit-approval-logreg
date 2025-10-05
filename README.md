# Credit Approval with Logistic Regression

## Run (synthetic demo)
```bash
python src/main.py --generate-synthetic --random-state 42 --balance --target-precision 0.80
```

## Run (your CSV)
```bash
python src/main.py --data-path data/credit.csv --target Approved --positive-class yes --balance --target-precision 0.85
```

Artifacts saved to `figures/` and `outputs/`.

### Notes
- Preprocessing via ColumnTransformer: median impute + scale (numeric), most-frequent + one-hot (categorical).
- LogisticRegression with configurable penalty/C; class_weight="balanced" optional.
- Coefficients + odds ratios saved to `outputs/coefficients_oddsratios.csv`.
- Metrics & plots saved (confusion matrix, ROC, PR, threshold tradeoff). Evaluation JSON in `outputs/evaluation.json`.
