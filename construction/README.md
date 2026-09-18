# FUTURE-SENSE construction materials

This supplement discloses the full future semantic-gap tree, the stage-1 routing
implementation, and the stage-2 candidate-sense generation prompt. The existing
`FUTURE-SENSE_v1.0.csv` remains the task-facing release of 403 curated instances.

## Contents

| Material | File |
| --- | --- |
| Full English tree used by routing | `trees/future_semantic_gap_tree.en.json` |
| Full Chinese/English node inventory and paths | `trees/future_semantic_gap_tree.bilingual.json` |
| Standalone construction router | `routing/route_wordnet.py` |
| Dependencies | `routing/requirements.txt` |
| Generation prompt template | `prompts/new_sense_generation.txt` |
| Prompt fields and recorded generation settings | `prompts/README.md` |

The tree contains 118 nodes: one root, nine broad domains, 27 intermediate
regions, and 81 fine-grained gaps. The bilingual inventory identifies every node
and its complete path. The router links each intermediate region with each of
its fine-grained children to obtain 81 routing leaves under nine domains.
English names and gap descriptions are used to form leaf embedding texts;
the original `lexical` fields are retained as part of the full tree but are not
used by this router to select candidate word forms or encode gap texts.

## Stage 1: routing

Use Python 3.10 or newer. Install dependencies and obtain WordNet 3.0:

```bash
python -m pip install -r construction/routing/requirements.txt
python -m nltk.downloader wordnet
python construction/routing/route_wordnet.py --device cpu
```

For a CUDA environment, use `--device cuda`. The sentence-transformers model
`sentence-transformers/all-mpnet-base-v2` is required and may be downloaded on
first use. A local model path can be supplied through `--model-name`. Model
weights and WordNet data are not bundled; their original terms apply.
The NLTK corpus should report `3.0` via `nltk.corpus.wordnet.get_version()`.
The dependency pins record the packaging environment rather than asserting the
exact versions used in the original run. Hardware/library differences and model
revisions can affect floating-point values and near-tied rankings.

The original settings are the defaults: at least four content-word senses
(`--min-senses 3` is exclusive), domain top-five aggregation (`--top-m 5`), and
five accepted word forms per leaf (`--min-per-leaf 5`). WordNet multiword lemmas
are allowed; `--single-word-only` changes that setting. Stopwords and tokens
without valid content are filtered using the original rules.

For each word, the router considers its recorded senses against **every** leaf.
Within each first-level domain, it averages the five largest sense-leaf pairing
scores; these need not involve five distinct leaves. Each specific leaf also
receives its maximum sense similarity and corresponding source-sense anchor.
All lemma-leaf options are ordered by descending domain score, descending leaf
score, first-domain ID, leaf ID, and lemma. The greedy allocator accepts a pair
only while the leaf has capacity and the lemma has not already been used.
Consequently, capacity constraints may place a word outside its individually
highest-scoring domain. There is no retrospective top-K rejection gate here.

The recorded complete construction used 8,712 candidate forms and 705,672
lemma-leaf options, producing 405 assignments (81 x 5). The later 403-instance
release follows generation and human curation; rerunning this router alone does
not regenerate that final dataset.

Outputs are written to `routing_outputs/` by default:

- `future_tree_three_layer_linked.json`: the collapsed tree, including exact
  leaf `document_text` and source-node links.
- `future_leaf_word_queues.csv`: accepted assignments, scores, and source senses.
- `future_leaf_word_queues.json`: the same queues with metadata.
- `future_leaf_collection_summary.json`: counts and completion status.

Use `--output-dir` and `--embedding-cache-dir` to select other locations. A small
smoke run can use `--max-lemmas 30 --min-per-leaf 1 --allow-incomplete`; it is not
the reported full construction. The script is a standalone extraction of the
original executed routing path and its essential utilities. Unused alternative
routes and surrounding workflow code have been omitted.

## Stage 2: generation prompt

See `prompts/README.md` for all template substitutions. The prompt takes textual
routing evidence, gap context, the selected old-sense anchor, and supplied
structured historical/current senses. It does not take embedding vectors.

The prompt exposes the candidate-generation instructions, not a complete
end-to-end dataset reconstruction pipeline. Per-instance historical evidence,
retrieval/structuring code, generation API orchestration, human-curation tools,
and evaluation code are outside this supplement. Other construction metadata
remain with the authors as described in the main release.

## License

These added project materials follow the repository's root `LICENSE`. This
supplement does not change the existing dataset or root license. Third-party
lexical resources, model weights, and software retain their respective terms.
