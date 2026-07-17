"""
Retrain the phishing detection model using XGBoost on the UCI dataset.
Now trains on 28 features (25 original UCI + 3 new features).

Run this from your project root:
    python RetrainXGboost.py

Requirements:
    - data/Training Dataset.arff   (UCI Phishing Websites Dataset)
    - src/data_loader.py
    - src/model.py
    - src/feature_extraction.py    (updated 28-feature version)

Output:
    - models/phishing_detector.pkl  (overwrites existing model)
    - models/metrics.json
"""

from backend_flask.model import PhishingDetector
from backend_flask.data_loader import DataLoader
import pandas as pd
import os
import sys
import json
import math
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


DATASET_PATH = "model_dataset/Training Dataset.arff"
MODEL_OUTPUT = "model_xgboost/phishing_detector.pkl"
METRICS_OUTPUT = "model_xgboost/metrics.json"

# -----------------------------------------------------------------------
# TLD and private-IP reference data — mirrors feature_extraction.py
# so the engineered columns are consistent between training and inference.
# -----------------------------------------------------------------------
SUSPICIOUS_TLDS = [
    '.xyz', '.click', '.club', '.online', '.site', '.lat',
    '.sbs', '.top', '.work', '.loan', '.win', '.gq', '.ml',
    '.cf', '.ga', '.tk', '.pw', '.cc', '.su', '.ws',
]

PRIVATE_IP_PATTERNS = [
    r'^192\.168\.', r'^10\.', r'^172\.(1[6-9]|2[0-9]|3[01])\.',
    r'^127\.', r'^0\.', r'^169\.254\.', r'^::1$', r'^localhost$',
]


def _is_private_ip(domain: str) -> float:
    for p in PRIVATE_IP_PATTERNS:
        if re.match(p, str(domain)):
            return -1.0
    return 1.0


def _url_entropy(url: str) -> float:
    url = str(url)
    if not url:
        return 0.0
    freq = {}
    for ch in url:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(url)
    entropy = -sum((c / length) * math.log2(c / length) for c in freq.values())
    if entropy < 3.5:
        return 1.0
    if entropy <= 4.5:
        return 0.0
    return -1.0


def _suspicious_tld(domain: str) -> float:
    domain = str(domain).lower()
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            return -1.0
    return 1.0


def add_new_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add the 3 new engineered features to a UCI feature DataFrame.

    The UCI dataset does not contain raw domain strings, so we derive
    the new features from existing UCI columns:

    is_private_ip:
        The UCI 'having_IP_Address' column already flags IP-based domains
        as -1.0. A private IP is a subset of that. We set is_private_ip=-1.0
        wherever having_IP_Address=-1.0 as the best available proxy.
        (During live inference, _is_private_ip() checks the actual domain
        string directly, which is more precise.)

    url_entropy:
        No raw URL string is available in the UCI dataset, so we derive
        a proxy from URL_Length:
          short URL  (1.0)  -> low entropy    -> 1.0
          medium URL (0.0)  -> medium entropy -> 0.0
          long URL  (-1.0)  -> high entropy   -> -1.0
        This keeps training and inference directionally consistent.

    suspicious_tld:
        No TLD string is in the UCI dataset. We use having_Sub_Domain as
        a proxy: more subdomains correlates with unusual domain structures
        including suspicious TLDs. This is a rough approximation; the
        live extractor checks the actual TLD directly.
    """
    X = X.copy()

    # is_private_ip: proxy from having_IP_Address
    if 'having_IP_Address' in X.columns:
        X['is_private_ip'] = X['having_IP_Address'].apply(
            lambda v: -1.0 if float(v) == -1.0 else 1.0
        )
    else:
        X['is_private_ip'] = 1.0

    # url_entropy: proxy from URL_Length
    if 'URL_Length' in X.columns:
        X['url_entropy'] = X['URL_Length'].apply(
            lambda v: float(v)   # UCI URL_Length is already 1.0/0.0/-1.0
        )
    else:
        X['url_entropy'] = 0.0

    # suspicious_tld: proxy from having_Sub_Domain
    if 'having_Sub_Domain' in X.columns:
        X['suspicious_tld'] = X['having_Sub_Domain'].apply(
            lambda v: -1.0 if float(v) == -1.0 else 1.0
        )
    else:
        X['suspicious_tld'] = 1.0

    return X


def main():
    # ------------------------------------------------------------------ #
    # 1. Check dataset                                                     #
    # ------------------------------------------------------------------ #
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at '{DATASET_PATH}'")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Load UCI data (25 features after dropping 5 discontinued ones)   #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Step 1: Loading UCI Phishing Dataset")
    print("=" * 60)

    loader = DataLoader()
    X, y = loader.load_uci_dataset(DATASET_PATH)

    print(
        f"\nUCI features loaded : {X.shape[1]} features, {X.shape[0]} samples")

    # ------------------------------------------------------------------ #
    # 3. Add 3 new engineered features -> 28 total                        #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Step 2: Adding 3 New Engineered Features")
    print("=" * 60)

    X = add_new_features(X)
    print(f"\nFeatures after engineering : {X.shape[1]}")
    print(f"New columns added          : is_private_ip, url_entropy, suspicious_tld")

    print(f"\nFull feature list ({X.shape[1]}):")
    for i, col in enumerate(X.columns, 1):
        marker = " <-- NEW" if col in ('is_private_ip',
                                       'url_entropy', 'suspicious_tld') else ""
        print(f"  {i:>2}. {col}{marker}")

    # ------------------------------------------------------------------ #
    # 4. Class distribution                                                #
    # ------------------------------------------------------------------ #
    counts = y.value_counts()
    print(f"\nClass distribution:")
    print(f"  Legitimate (0): {counts.get(0, 0)}")
    print(f"  Phishing   (1): {counts.get(1, 0)}")

    # ------------------------------------------------------------------ #
    # 5. Train XGBoost                                                     #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Step 3: Training XGBoost Classifier (28 features)")
    print("=" * 60)

    detector = PhishingDetector(model_type='xgboost', random_state=42)
    val_metrics = detector.train(X, y, validation_split=0.2)

    print("\nValidation Metrics (20% holdout):")
    print(f"  Accuracy  : {val_metrics['accuracy']*100:.2f}%")
    print(f"  Precision : {val_metrics['precision']*100:.2f}%")
    print(f"  Recall    : {val_metrics['recall']*100:.2f}%")
    print(f"  F1 Score  : {val_metrics['f1_score']*100:.2f}%")
    print(f"  ROC-AUC   : {val_metrics['roc_auc']*100:.2f}%")

    # ------------------------------------------------------------------ #
    # 6. Cross-validation                                                  #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Step 4: 5-Fold Cross-Validation")
    print("=" * 60)

    cv_metrics = detector.cross_validate(X, y, cv=5)
    print(f"\n  CV Accuracy: {cv_metrics['cv_mean_accuracy']*100:.2f}%"
          f" (+/- {cv_metrics['cv_std_accuracy']*100:.2f}%)")

    # ------------------------------------------------------------------ #
    # 7. Feature importance                                                #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Step 5: Top 15 Feature Importances")
    print("=" * 60)

    importances = detector.get_feature_importances()
    top15 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"\n{'Rank':<5} {'Feature':<35} {'Importance'}")
    print("-" * 58)
    for rank, (feat, imp) in enumerate(top15, 1):
        bar = "█" * int(imp * 40)
        marker = " <-- NEW" if feat in ('is_private_ip',
                                        'url_entropy', 'suspicious_tld') else ""
        print(f"{rank:<5} {feat:<35} {imp:.4f}  {bar}{marker}")

    # ------------------------------------------------------------------ #
    # 8. Save                                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Step 6: Saving Model")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)
    detector.save(MODEL_OUTPUT)
    print(f"\nModel saved    : {MODEL_OUTPUT}")
    print(f"Feature count  : {X.shape[1]}")
    print(f"Feature names  : {list(X.columns)}")

    all_metrics = {**val_metrics, **cv_metrics, "feature_count": X.shape[1]}
    with open(METRICS_OUTPUT, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved  : {METRICS_OUTPUT}")

    # ------------------------------------------------------------------ #
    # 9. Summary                                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start Flask:  python -m backend_flask.app")
    print("  2. Start Gradio: python -m frontend_gradio.gradio_app")
    print("  3. Open the local host link on browser")


if __name__ == "__main__":
    main()
