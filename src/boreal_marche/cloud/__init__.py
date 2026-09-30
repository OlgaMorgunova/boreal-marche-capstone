"""Hyper-parameter search backends: ``local`` (Optuna), ``aws`` (SageMaker), ``azure`` (Azure ML).

All three run the *same* training entry point (``boreal_marche.retention.sm_train``) on the same
gold table, so their scores are comparable. Cloud SDKs are imported lazily: they are only needed by
whoever actually launches a cloud job.
"""
