# Local model

Place a trusted `spam_pipeline.joblib` here, or run `train_model.py` with your own datasets. The pipeline must use labels 0 (ham) and 1 (spam) and expose `predict_proba`.

The original model was trained on private SMS data and is not published. `metrics.json` contains the aggregate results recorded during the original training run; it is not a new evaluation of your replacement model. Retraining replaces both output files.

Do not commit raw messages or a private-data-derived model without checking whether they can be shared. Joblib files can execute code when loaded; use only files from trusted sources.
