# FUTURE-SENSE v1.0

FUTURE-SENSE v1.0 is the task-facing evaluation release of a dataset for evaluating lexical-semantic generalization over candidate novel senses. It contains 403 human-curated instances.

## Data Fields

- `instance_id`: unique identifier for each evaluation instance.
- `word`: target word form.
- `example`: contextual sentence containing the candidate novel sense.
- `meaning`: reference definition of the candidate novel sense.

## Task Setup

For the definition-recovery task, `word` and `example` are provided as model inputs, while `meaning` serves as the reference answer.

## Release Scope

This task-facing release contains only the fields required for evaluation. Additional construction and traceability metadata are retained by the authors and are not included in the public release.

The data are provided solely for non-commercial academic research.

## Citation

Citation details will be updated upon publication.
