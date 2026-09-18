# Stage-2 new-sense generation prompt

`new_sense_generation.txt` is extracted from the `build_prompt` function of the
recorded DeepSeek V4 Pro prompt-v2 construction implementation. Its instruction
text and output schema are preserved; only Python interpolation expressions
are replaced with named `${...}` placeholders. It is not a shortened paraphrase.
The original heading says "Five-Step" although the list contains eight steps;
that wording is retained rather than silently edited. The prompt's 2027–2030
horizon is also retained and is narrower than some horizons in the full tree.

## Substitution fields

| Placeholder | Value |
| --- | --- |
| `${predictions}` | `1` in the reported construction |
| `${word}` | Target word form, preserved exactly |
| `${routing_evidence_json}` | The assignment CSV row as JSON, preserving field order and CSV string values |
| `${future_context_json}` | The context object described below |
| `${nearest_sense_id}` | Selected WordNet synset ID |
| `${nearest_sense_gloss}` | Selected old-sense gloss |
| `${history_source_word}` | Word form providing the historical sequence |
| `${history_source_relation_json}` | JSON describing any source/base-form relationship |
| `${historical_sequence}` | Up to the first 12 structured history entries, rendered as below |
| `${target_wordnet_senses_json}` | Target-word current senses, as supplied to the prompt |
| `${source_wordnet_senses_json}` | Source-word current senses, or `[]` when absent |
| `${leaf_id}` | Assigned linked-leaf ID |

All JSON placeholders use the equivalent of `json.dumps(value,
ensure_ascii=False, indent=2)`. Preserve the literal JSON braces in the output
schema. In Python, `string.Template(...).substitute(values)` handles this
placeholder syntax without interpreting those braces.

The context object uses this field order:

```text
first_id, first_name, first_gap,
leaf_id, leaf_name, leaf_horizon, leaf_gap,
source_layer3_name, source_layer4_name,
nearest_sense_id, nearest_sense_gloss, leaf_score, gate_score
```

Values come from the accepted routing row and collapsed tree. Names use the
CSV value when non-empty, otherwise the tree value; domain gap and leaf horizon
come from their nodes. `leaf_gap` is the collapsed node's `gap`, which combines
the intermediate and fine-grained descriptions. Scores retain their CSV string
form in this context object. These context values are necessary prompt inputs;
the code that assembles them is not included.

Each history entry is rendered on one line:

```text
{id} | {time} | {historical_sense} | WordNet={is_wordnet_sense} {wordnet_match_type} {matched_wordnet_ids}
```

The list field uses Python list string representation in this history block,
not JSON. The original defaults for absent fields are `T?`, `N/A`, an empty
historical sense, `None` for the flag, an empty match type, and `[]` respectively.
An absent sequence renders `NO HISTORICAL SEQUENCE AVAILABLE`. Current-sense
entries contain `id`, `definition`, and `examples` (at most two examples per
sense); the ordinary compaction step retains up to 12 senses. Supply the actual
structured evidence, rather than fabricating historical entries from the gap.

## Recorded main-run settings

- Model identifier: `deepseek-v4-pro` through the official DeepSeek API.
- One user message containing the filled prompt; no separate system message.
- One candidate prediction per routed assignment.
- `thinking` enabled; `reasoning_effort` set to `max`.
- JSON-object response format; maximum output tokens: 9,000.

These settings describe the stored main-run configuration, not a promise that
the same endpoint remains available. The output JSON schema is included in the
template. The generated definitions, examples and boundaries are provisional;
the released dataset additionally underwent human curation.
