"""The interaction classifier, Layer A: every 10 s window of a session's fused table is labelled
individual, social or collaborative, with calibrated probabilities; the binary target (any
interaction) is always derived as p(social) + p(collaborative), never trained on its own. Layer B,
the 60 s role function of each person, builds on it and is not here.

What reads and writes where:
- in: artifacts/<session>/analysis/features/<session>_window_features.csv (mmla ses-fuse) and
  artifacts/<session>/labels/<coder>.jsonl (mmla ses-code);
- layout: the roster, the tokens (group, persons, pairs), their scaling, the 82-column pooled view
  and its lags; labels: reading and joining the coders' labels; splits: TEST and the
  leave-one-lesson-out folds;
- out, mmla ses-classify: artifacts/_analysis/interaction/<run>/ (predictions.csv, metrics.json,
  per_session.csv, confusion.csv, label_counts.csv, roster.json, data_checks.json, config.json);
  the _analysis prefix keeps ses-code's glob('exp_*') from taking it for a session;
- out, mmla ses-jev: its request cache under artifacts/_analysis/interaction/jev/cache/ and, per
  session, artifacts/<session>/analysis/interaction/jev_<variant>.jsonl.

Transcript text never leaves the machine, and no feature carries a tag id or a session id.
"""
