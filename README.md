# FUTURE-SENSE v1.0

FUTURE-SENSE is a resource for evaluating lexical-semantic generalization over candidate novel senses. The v1.0 evaluation dataset contains 403 human-curated instances with unique target word forms. This repository also provides the complete future semantic-gap tree, stage-1 routing code, and stage-2 candidate-sense generation prompt.

## Repository Contents

| Material | Location |
| --- | --- |
| Evaluation dataset: 403 instances | [`FUTURE-SENSE_v1.0.csv`](FUTURE-SENSE_v1.0.csv) |
| Construction overview and usage | [`construction/README.md`](construction/README.md) |
| Complete English semantic-gap tree | [`construction/trees/future_semantic_gap_tree.en.json`](construction/trees/future_semantic_gap_tree.en.json) |
| Chinese/English node inventory and paths | [`construction/trees/future_semantic_gap_tree.bilingual.json`](construction/trees/future_semantic_gap_tree.bilingual.json) |
| Stage-1 routing implementation | [`construction/routing/route_wordnet.py`](construction/routing/route_wordnet.py) |
| Stage-2 generation prompt | [`construction/prompts/new_sense_generation.txt`](construction/prompts/new_sense_generation.txt) |
| License | [`LICENSE`](LICENSE) |

## Data Fields

- `instance_id`: unique identifier for each evaluation instance.
- `word`: target word form.
- `example`: contextual sentence containing the candidate novel sense.
- `meaning`: reference definition of the candidate novel sense.

## Task Setup

For the definition-recovery task, `word` and `example` are provided as model inputs, while `meaning` serves as the reference answer.

## Construction Overview

The semantic-gap tree contains one root, nine broad domains, 27 intermediate regions, and 81 fine-grained gaps. For routing, each intermediate region is combined with its fine-grained child description to form a linked leaf.

Stage 1 scores all candidate word-form–leaf pairs using WordNet senses and MPNet embeddings. It aggregates the top five sense–leaf similarities within each domain and uses the maximum sense similarity for each specific leaf. All pairs are globally ranked by domain score, leaf score, and deterministic tie-breaking keys. Greedy allocation assigns each word form at most once and fills five slots per leaf. The recorded run used 8,712 candidate forms and produced 405 assignments.

Stage 2 conditions candidate-sense generation on the assigned gap, source-sense anchor, structured historical evidence, and current recorded senses. Of the 405 assignments, 403 yielded non-empty candidate records; human curation produced the final evaluation release. The prompt exposes the generation instructions and output schema. See the [construction documentation](construction/README.md) for routing dependencies and commands.

## Release Scope and Reproducibility

The evaluation CSV contains the four task-facing fields listed above. The construction supplement adds the complete future tree, a standalone routing implementation, and the candidate-generation prompt. The v1.0 CSV is unchanged by this supplement.

Other per-instance construction and traceability metadata are retained by the authors. Historical evidence retrieval and structuring code, context assembly code, generation API orchestration, human-curation tools, and evaluation code are outside this release. The supplement is not a complete end-to-end reconstruction pipeline for the final curated dataset.

## License

The dataset and added project materials are provided under the [Creative Commons Attribution 4.0 International license](LICENSE). Third-party lexical resources, model weights, and software retain their respective terms.

## Citation

Citation details will be updated upon publication.
