# Trained model

`spam_pipeline.joblib` bundles the TF-IDF vectorizer and Logistic Regression classifier used by the demo. Labels are 0 (ham) and 1 (spam). The server loads this file automatically.

`metrics.json` contains the aggregate results recorded during the original training run. The source SMS datasets are not included. See the main README for the evaluation split, results, and limitations.

Run `train_model.py` with your own datasets to replace both files. Only load trusted joblib files: Python model serialization can execute code when loading a file.
