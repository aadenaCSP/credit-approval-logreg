#!/usr/bin/env python
"""
Week 2 — Credit Approval with Logistic Regression

Run examples:
  Synthetic:
    python src/main.py --generate-synthetic --random-state 42 --balance --target-precision 0.80

  CSV:
    python src/main.py --data-path data/credit.csv --target Approved --positive-class yes --balance --target-precision 0.85
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)

FIG_DIR, OUT_DIR = Path("figures"), Path("outputs")
FIG_DIR.mkdir(exist_ok=True); OUT_DIR.mkdir(exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser(description="Credit approval logistic regression pipeline")
    # Data
    p.add_argument("--data-path", type=str, default=None, help="CSV/XLSX path. If omitted, use --generate-synthetic.")
    p.add_argument("--sheet", type=str, default=None, help="Excel sheet name if needed.")
    p.add_argument("--target", type=str, default="approved", help="Target column name.")
    p.add_argument("--positive-class", type=str, default=None, help="Label considered positive if target is categorical.")
    p.add_argument("--drop-cols", type=str, default="", help="Comma-separated columns to drop (ids/leaks).")
    p.add_argument("--sensitive-cols", type=str, default="", help="Comma-separated sensitive columns to exclude.")
    # Synthetic
    p.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic dataset.")
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--imbalance", type=float, default=0.35, help="Approx positive class rate.")
    p.add_argument("--random-state", type=int, default=42)
    # Split/Model
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--balance", action="store_true", help="Use class_weight='balanced'.")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--penalty", type=str, choices=["l2", "l1"], default="l2")
    p.add_argument("--max-iter", type=int, default=1000)
    # Thresholding
    p.add_argument("--threshold", type=float, default=None, help="Manual threshold in [0,1].")
    p.add_argument("--target-precision", type=float, default=None, help="Pick smallest threshold achieving this precision.")
    return p.parse_args()

def load_dataframe(path: str, sheet: Optional[str]) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)

def generate_synthetic(n_samples: int, imbalance: float, random_state: int) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples, n_features=6, n_informative=4, n_redundant=0,
        n_clusters_per_class=2, weights=[1-imbalance, imbalance], class_sep=1.5,
        flip_y=0.02, random_state=random_state
    )
    df = pd.DataFrame(X, columns=["income_z","credit_z","dti_z","age_z","util_z","hist_z"])
    rng = np.random.default_rng(random_state)
    df["income"] = np.exp(df["income_z"]) * 30000
    df["credit_score"] = (df["credit_z"]*80 + 680).clip(300, 850)
    df["debt_to_income"] = (abs(df["dti_z"])*0.15 + 0.10).clip(0, 0.95)
    df["age"] = (abs(df["age_z"])*12 + 30).clip(18, 90)
    df["utilization"] = (abs(df["util_z"])*0.25 + 0.15).clip(0, 1.0)
    df["history_length_years"] = (abs(df["hist_z"])*3 + 6).clip(0, 40)
    emp = ["salaried","self-employed","contract","unemployed"]
    res = ["own","rent","mortgage"]
    states = ["CA","TX","NY","FL","IL"]
    df["employment_type"] = rng.choice(emp, size=n_samples, p=[0.6,0.15,0.2,0.05])
    df["residential_status"] = rng.choice(res, size=n_samples, p=[0.35,0.4,0.25])
    df["state"] = rng.choice(states, size=n_samples)
    for col in ["income","credit_score","debt_to_income","employment_type","residential_status"]:
        mask = rng.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
    df["approved"] = y.astype(int)
    df.drop(columns=[c for c in df.columns if c.endswith("_z")], inplace=True)
    return df

def coerce_target_to_binary(y: pd.Series, positive_class: Optional[str]) -> np.ndarray:
    if y.dtype.kind in "ifu":
        if set(np.unique(y)) <= {0,1}: return y.astype(int).to_numpy()
        return (y > np.median(y)).astype(int).to_numpy()
    if positive_class is None:
        classes = sorted(y.astype(str).unique().tolist())
        return (y.astype(str) == classes[-1]).astype(int).to_numpy()
    return (y.astype(str) == str(positive_class)).astype(int).to_numpy()

def infer_column_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    features = [c for c in df.columns if c != target]
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols

def feature_names_from_ct(ct: ColumnTransformer) -> List[str]:
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder": continue
        if isinstance(trans, Pipeline):
            if "onehot" in trans.named_steps:
                ohe = trans.named_steps["onehot"]
                names.extend(list(ohe.get_feature_names_out(cols)))
            else:
                names.extend(cols)
        else:
            names.extend(cols)
    return names

def plot_missingness(df: pd.DataFrame, out_path: Path):
    miss = df.isna().sum().sort_values(ascending=False)
    plt.figure(); miss.plot(kind="bar"); plt.title("Missing values per column"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_numeric_histograms(df: pd.DataFrame, numeric: List[str], out_dir: Path):
    for col in numeric:
        plt.figure(); plt.hist(df[col].dropna(), bins=30)
        plt.title(f"Histogram: {col}"); plt.xlabel(col); plt.ylabel("Frequency")
        plt.tight_layout(); plt.savefig(out_dir / f"hist_{col}.png"); plt.close()

def plot_correlation(df: pd.DataFrame, numeric: List[str], out_path: Path):
    if not numeric: return
    corr = df[numeric].corr(numeric_only=True)
    plt.figure(); plt.imshow(corr, interpolation="nearest")
    plt.title("Correlation heatmap (numeric)")
    plt.xticks(range(len(numeric)), numeric, rotation=90); plt.yticks(range(len(numeric)), numeric)
    plt.colorbar(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_confusion(cm: np.ndarray, labels: List[str], out_path: Path):
    plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix")
    plt.xticks([0,1],[f"Pred {l}" for l in labels]); plt.yticks([0,1],[f"True {l}" for l in labels])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def choose_threshold(y_true: np.ndarray, y_proba: np.ndarray, target_precision: Optional[float]=None):
    ps, rs, ts = precision_recall_curve(y_true, y_proba)
    ts = np.r_[0.0, ts]
    if target_precision is not None:
        for t, p in zip(ts, ps):
            if p >= target_precision:
                return float(t), {"rule":"precision>=target","target_precision":target_precision}
    f1s = (2*ps*rs)/np.clip(ps+rs,1e-12,None)
    best_idx = int(np.nanargmax(f1s))
    return float(ts[best_idx]), {"rule":"max_f1","best_f1":float(f1s[best_idx])}

def save_threshold_tradeoff(y_true: np.ndarray, y_proba: np.ndarray, out_path: Path):
    ps, rs, ts = precision_recall_curve(y_true, y_proba)
    ts = np.r_[0.0, ts]
    plt.figure(); plt.plot(ts, ps, label="Precision"); plt.plot(ts, rs, label="Recall")
    plt.xlabel("Threshold"); plt.ylabel("Score"); plt.title("Threshold trade-off"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])

def build_pipeline(preprocessor: ColumnTransformer, C: float, penalty: str, balanced: bool, max_iter: int) -> Pipeline:
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    class_weight = "balanced" if balanced else None
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, class_weight=class_weight)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

def main():
    args = parse_args()

    # Load or synthesize data
    if args.generate_synthetic:
        df = generate_synthetic(args.n_samples, args.imbalance, args.random_state)
        dataset_note = {"type":"synthetic","n_samples":args.n_samples,"imbalance_positive_rate":args.imbalance,"random_state":args.random_state}
    else:
        if not args.data_path:
            raise SystemExit("Provide --data-path or use --generate-synthetic.")
        df = load_dataframe(args.data_path, args.sheet)
        dataset_note = {"type":"csv_or_excel","path":args.data_path,"sheet":args.sheet}

    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not found. Available: {list(df.columns)}")

    # Drop ids/sensitive
    to_drop = []
    if args.drop_cols: to_drop += [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    if args.sensitive_cols: to_drop += [c.strip() for c in args.sensitive_cols.split(",") if c.strip()]
    to_drop = [c for c in to_drop if c in df.columns]
    if to_drop:
        print(f"[INFO] Dropping columns: {to_drop}")
        df = df.drop(columns=to_drop)

    # Target & features
    y_raw = df[args.target]
    X = df.drop(columns=[args.target]).copy()
    y = coerce_target_to_binary(y_raw, args.positive_class)

    # Column types
    full = pd.concat([X, pd.Series(y, name=args.target)], axis=1)
    num_cols, cat_cols = infer_column_types(full, args.target)

    # Targeted EDA
    pos_rate = float(np.mean(y))
    print(f"[EDA] Positive rate = {pos_rate:.3f} ({np.sum(y)}/{len(y)})")
    plot_missingness(full, FIG_DIR / "missingness.png")
    plot_numeric_histograms(X, num_cols, FIG_DIR)
    plot_correlation(pd.concat([X[num_cols], pd.Series(y, name=args.target)], axis=1), num_cols, FIG_DIR / "corr_numeric.png")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size,
                                                        random_state=args.random_state, stratify=y)

    # Pipeline
    pre = build_preprocessor(num_cols, cat_cols)
    pipe = build_pipeline(pre, args.C, args.penalty, args.balance, args.max_iter)
    pipe.fit(X_train, y_train)

    # Interpretation
    ct = pipe.named_steps["preprocessor"]
    model: LogisticRegression = pipe.named_steps["model"]
    feat_names = feature_names_from_ct(ct)
    coefs = model.coef_.ravel()
    intercept = model.intercept_[0]
    odf = pd.DataFrame({
        "feature": feat_names,
        "coef": coefs,
        "odds_ratio": np.exp(coefs),
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)
    odf.to_csv(OUT_DIR / "coefficients_oddsratios.csv", index=False)
    print("\n[INTERPRETATION] Top 10 | feature, coef, OR")
    print(odf.head(10)[["feature","coef","odds_ratio"]].to_string(index=False))
    print(f"[INTERPRETATION] Intercept log-odds={intercept:.4f} OR={np.exp(intercept):.4f}")

    # Predictions & metrics
    y_proba = pipe.predict_proba(X_test)[:, 1]
    if args.threshold is not None:
        thresh, rule = float(args.threshold), {"rule":"user_supplied"}
    else:
        thresh, rule = choose_threshold(y_test, y_proba, args.target_precision)

    y_pred = (y_proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    rocauc = roc_auc_score(y_test, y_proba)
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(pr_rec, pr_prec)

    # Plots
    plot_confusion(cm, ["0","1"], FIG_DIR / "confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(); plt.plot(fpr, tpr, label=f"ROC AUC={rocauc:.3f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "roc_curve.png"); plt.close()
    plt.figure(); plt.plot(pr_rec, pr_prec, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "pr_curve.png"); plt.close()
    save_threshold_tradeoff(y_test, y_proba, FIG_DIR / "threshold_tradeoff.png")

    # Save eval JSON
    eval_payload = {
        "dataset": dataset_note,
        "dropped_columns": to_drop,
        "random_state": args.random_state,
        "split": {"test_size": args.test_size, "stratified": True},
        "model": {
            "type": "LogisticRegression",
            "penalty": args.penalty,
            "C": args.C,
            "class_weight": "balanced" if args.balance else None,
            "max_iter": args.max_iter
        },
        "threshold_rule": rule,
        "threshold": thresh,
        "metrics": {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": rocauc, "pr_auc": pr_auc,
            "confusion_matrix": cm.tolist()
        }
    }
    with open(OUT_DIR / "evaluation.json", "w") as f:
        json.dump(eval_payload, f, indent=2)

    # Console summary
    print(f"\\n[EVALUATION @ threshold={thresh:.3f}]")
    print(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} ROC-AUC={rocauc:.3f} PR-AUC={pr_auc:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:\\n", cm)

    # Responsible modeling note
    if args.sensitive_cols:
        print("\\n[RESPONSIBLE MODELING] Sensitive columns were dropped:", args.sensitive_cols)
        print("Consider proxy risk (e.g., location, tenure) and document exclusions.")

    print("\\nArtifacts saved to ./figures and ./outputs")
    print("Done.")

if __name__ == "__main__":
    main()
