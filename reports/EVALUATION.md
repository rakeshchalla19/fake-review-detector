# Evaluation status

The Amazon Reviews ’23 training source has no fake/genuine ground-truth labels, so supervised classification metrics cannot be computed. The active Isolation Forest is an unsupervised outlier ranker, not a fake-review classifier.

Its 5% contamination setting creates a review shortlist relative to the training sample. It does not mean 5% of reviews are fraudulent. Review `reports/metrics.json` for the exact training sample count, source checksum, features, and threshold note.

An external evaluation would require independently verified fake and genuine labels, with a held-out split by product and reviewer. Until such data is available, use results to inspect unusual patterns only; do not treat them as verdicts.
