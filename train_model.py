"""Train the SMS classifier and save validation metrics and the fitted pipeline."""
import argparse
from pathlib import Path
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

from data_utils import DATASET_FILES, load_combined_corpus

PROJECT_DIR = Path(__file__).resolve().parent
OUT_MODEL = PROJECT_DIR / 'model' / 'spam_pipeline.joblib'
OUT_METRICS = PROJECT_DIR / 'model' / 'metrics.json'


def build_pipeline():
    # Balance the class weights to give both labels equal weight during fitting.
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', min_df=1)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', type=Path, help='Folder containing the CSVs listed in data_utils.DATASET_FILES')
    args = parser.parse_args()
    missing = [name for name in DATASET_FILES if not (args.data_dir / name).is_file()]
    if missing:
        parser.error('Missing dataset files: ' + ', '.join(missing))
    print(f'Loading and combining datasets from {args.data_dir} ...')
    df = load_combined_corpus(args.data_dir)
    print(f'Combined corpus: {len(df)} unique messages '
          f'({int(df.label.sum())} spam / {int((1 - df.label).sum())} ham)')

    X = df['message_clean'].values
    y = df['label'].values

    # Evaluate on a stratified split before fitting the distributed model.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)

    val_pipe = build_pipeline()
    val_pipe.fit(X_train, y_train)
    y_pred = val_pipe.predict(X_val)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    metrics = {
        'validation_split': '85/15 (stratified, random_state=42)',
        'n_train': int(len(X_train)),
        'n_val': int(len(X_val)),
        'accuracy': round(accuracy_score(y_val, y_pred), 4),
        'precision': round(precision_score(y_val, y_pred, zero_division=0), 4),
        'recall_sensitivity': round(recall_score(y_val, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_val, y_pred, zero_division=0), 4),
        'specificity': round(tn / (tn + fp), 4) if (tn + fp) > 0 else None,
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
    }
    print('\nHeld-out validation metrics:')
    print(json.dumps(metrics, indent=2))

    # Fit the distributed model on the complete corpus after evaluation.
    print('\nRetraining final pipeline on the full dataset for deployment...')
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, OUT_MODEL)
    with open(OUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nSaved model -> {OUT_MODEL}')
    print(f'Saved metrics -> {OUT_METRICS}')


if __name__ == '__main__':
    main()
