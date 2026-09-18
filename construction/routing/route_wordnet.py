#!/usr/bin/env python3
"""Stage 1: WordNet-to-gap routing used in FUTURE-SENSE construction.

Collapse the full tree into root -> domain -> linked gap. Score all lemma-gap
pairs by domain top-5 sense-gap mean and local maximum similarity, then scan
the global ordering with one assignment per lemma and five slots per leaf.
This is the construction allocator, not the historical top-K gate backtest.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TREE_JSON = CONSTRUCTION_DIR / "trees" / "future_semantic_gap_tree.en.json"
DEFAULT_OUTPUT_DIR = Path("routing_outputs")
DEFAULT_CACHE_DIR = Path("routing_embedding_cache")

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[\u4e00-\u9fff]{2,}")

Vector = Union[Dict[str, float], np.ndarray]

@dataclass(frozen=True)
class Sense:
    synset_id: str
    pos: str
    offset: str
    lemma_names: Tuple[str, ...]
    gloss: str

    @property
    def text(self) -> str:
        return " ".join([self.synset_id, " ".join(self.lemma_names), self.gloss])

@dataclass
class LemmaEntry:
    lemma: str
    senses: List[Sense]

    @property
    def prototype_text(self) -> str:
        return " ".join(sense.text for sense in self.senses)

def tokenize(text: str) -> List[str]:
    tokens = []
    normalized = text.replace("_", " ").replace("-", " ")
    for token in TOKEN_RE.findall(normalized.lower()):
        if token in STOPWORDS:
            continue
        if len(token) == 1 and token.isascii():
            continue
        tokens.append(token)
    return tokens

def cosine(vec_a: Vector, vec_b: Vector) -> float:
    if isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        if vec_a.size == 0 or vec_b.size == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b))

    if not isinstance(vec_a, dict) or not isinstance(vec_b, dict):
        raise TypeError("cosine expects two sparse dict vectors or two dense numpy vectors")

    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    return sum(value * vec_b.get(token, 0.0) for token, value in vec_a.items())

class WordNetLoadError(RuntimeError):
    pass

def load_wordnet_from_nltk() -> List[LemmaEntry]:
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception as exc:
        raise WordNetLoadError("NLTK is not installed") from exc

    try:
        lemma_names = sorted(wn.all_lemma_names())
    except LookupError as exc:
        raise WordNetLoadError("NLTK is installed but the WordNet corpus is missing") from exc

    entries: List[LemmaEntry] = []
    for lemma in lemma_names:
        senses = []
        for synset in wn.synsets(lemma):
            examples = " ".join(synset.examples())
            gloss = " ".join([synset.definition(), examples]).strip()
            senses.append(
                Sense(
                    synset_id=synset.name(),
                    pos=synset.pos(),
                    offset=str(synset.offset()).zfill(8),
                    lemma_names=tuple(synset.lemma_names()),
                    gloss=gloss,
                )
            )
        if senses:
            entries.append(LemmaEntry(lemma=lemma, senses=senses))
    return entries

def read_data_file(data_path: Path, pos: str) -> Dict[str, Sense]:
    senses: Dict[str, Sense] = {}
    with data_path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("  "):
                continue
            if " | " in line:
                data_part, gloss = line.split(" | ", 1)
            else:
                data_part, gloss = line, ""
            parts = data_part.split()
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            offset = parts[0]
            try:
                word_count = int(parts[3], 16)
            except ValueError:
                continue
            lemma_names = []
            cursor = 4
            for _ in range(word_count):
                if cursor + 1 >= len(parts):
                    break
                lemma_names.append(parts[cursor])
                cursor += 2
            synset_id = f"{offset}-{pos}"
            senses[offset] = Sense(
                synset_id=synset_id,
                pos=pos,
                offset=offset,
                lemma_names=tuple(lemma_names),
                gloss=gloss,
            )
    return senses

def read_index_file(index_path: Path, pos: str, data_senses: Dict[str, Sense]) -> List[LemmaEntry]:
    entries = []
    with index_path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(" "):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            lemma = parts[0]
            try:
                ptr_count = int(parts[3])
            except ValueError:
                continue
            offsets_start = 6 + ptr_count
            offsets = parts[offsets_start:]
            senses = [data_senses[offset] for offset in offsets if offset in data_senses]
            if senses:
                entries.append(LemmaEntry(lemma=lemma, senses=senses))
    return entries

def load_wordnet_from_dict(wordnet_dir: Path) -> List[LemmaEntry]:
    dict_dir = wordnet_dir / "dict" if (wordnet_dir / "dict").is_dir() else wordnet_dir
    pos_files = {
        "n": ("index.noun", "data.noun"),
        "v": ("index.verb", "data.verb"),
        "a": ("index.adj", "data.adj"),
        "r": ("index.adv", "data.adv"),
    }

    merged: Dict[str, List[Sense]] = defaultdict(list)
    found_any = False
    for pos, (index_name, data_name) in pos_files.items():
        index_path = dict_dir / index_name
        data_path = dict_dir / data_name
        if not index_path.exists() or not data_path.exists():
            continue
        found_any = True
        data_senses = read_data_file(data_path, pos)
        for entry in read_index_file(index_path, pos, data_senses):
            merged[entry.lemma].extend(entry.senses)

    if not found_any:
        raise WordNetLoadError(
            f"No WordNet dict files found under {wordnet_dir}. "
            "Expected index.noun/data.noun etc."
        )

    return [LemmaEntry(lemma=lemma, senses=senses) for lemma, senses in sorted(merged.items())]

def load_wordnet(wordnet_dir: Optional[Path]) -> List[LemmaEntry]:
    if wordnet_dir:
        return load_wordnet_from_dict(wordnet_dir)
    return load_wordnet_from_nltk()

def normalize_dense(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms

def embedding_cache_key(model_name: str, keys: Sequence[str], texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for key, text in zip(keys, texts):
        digest.update(b"\0")
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:20]

def encode_with_sentence_transformers(
    keys: Sequence[str],
    texts: Sequence[str],
    model_name: str,
    cache_dir: Path,
    cache_label: str,
    batch_size: int,
    device: str,
) -> Dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = embedding_cache_key(model_name, keys, texts)
    cache_path = cache_dir / f"{cache_label}.{cache_key}.npz"

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_keys = cached["keys"].tolist()
        embeddings = cached["embeddings"]
        return {key: embeddings[i] for i, key in enumerate(cached_keys)}

    requested_key_set = set(keys)
    for existing_cache_path in sorted(cache_dir.glob(f"{cache_label}.*.npz")):
        cached = np.load(existing_cache_path, allow_pickle=False)
        cached_keys = cached["keys"].tolist()
        if requested_key_set.issubset(cached_keys):
            key_to_idx = {key: idx for idx, key in enumerate(cached_keys)}
            embeddings = cached["embeddings"]
            subset_embeddings = np.stack([embeddings[key_to_idx[key]] for key in keys])
            np.savez_compressed(cache_path, keys=np.array(keys), embeddings=subset_embeddings)
            return {key: subset_embeddings[i] for i, key in enumerate(keys)}

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required for --embedding-backend sentence-transformers. "
            "Install it with: python -m pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")
    embeddings = normalize_dense(embeddings).astype("float32")
    np.savez_compressed(cache_path, keys=np.array(keys), embeddings=embeddings)
    return {key: embeddings[i] for i, key in enumerate(keys)}

CONTENT_POSES = {"n", "v", "a", "s", "r"}

@dataclass(frozen=True)
class LeafOption:
    lemma: str
    sense_count: int
    gate_score: float
    leaf_score: float
    first_id: str
    first_name: str
    leaf_id: str
    leaf_name: str
    nearest_sense_id: str
    nearest_sense_gloss: str

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def safe_id(*parts: str) -> str:
    raw = "__".join(part for part in parts if part)
    raw = re.sub(r"[^A-Za-z0-9_:-]+", "-", raw)
    return re.sub(r"-+", "-", raw).strip("-").lower()

def field_text(node: dict) -> str:
    return " ".join(str(node.get(key, "")).strip() for key in ("name", "gap") if node.get(key))

def linked_leaf_doc(layer3: dict, layer4: Optional[dict]) -> str:
    if layer4 is None:
        return field_text(layer3)
    parts = [
        f"Subdomain: {layer3.get('name', '')}",
        str(layer3.get("gap", "")),
        f"Semantic gap: {layer4.get('name', '')}",
        str(layer4.get("gap", "")),
    ]
    return " ".join(part.strip() for part in parts if str(part).strip())

def collapse_future_tree(root: dict) -> dict:
    """Make root -> first-domain -> linked layer3/layer4 leaf tree."""
    collapsed_root = {
        key: value
        for key, value in root.items()
        if key != "children"
    }
    collapsed_root["depth"] = 0
    collapsed_root["children"] = []

    for first in root.get("children", []) or []:
        collapsed_first = {
            key: value
            for key, value in first.items()
            if key != "children"
        }
        collapsed_first["depth"] = 1
        collapsed_first["parent_id"] = collapsed_root["id"]
        collapsed_first["children"] = []

        for layer3 in first.get("children", []) or []:
            layer4_children = layer3.get("children", []) or []
            if not layer4_children:
                leaf_id = safe_id(layer3.get("id", ""))
                collapsed_first["children"].append(
                    {
                        "id": leaf_id,
                        "parent_id": collapsed_first["id"],
                        "depth": 2,
                        "name": layer3.get("name", ""),
                        "kind": layer3.get("kind", ""),
                        "horizon": layer3.get("horizon", ""),
                        "gap": layer3.get("gap", ""),
                        "document_text": linked_leaf_doc(layer3, None),
                        "source_layer3_id": layer3.get("id", ""),
                        "source_layer3_name": layer3.get("name", ""),
                        "source_layer4_id": "",
                        "source_layer4_name": "",
                    }
                )
                continue

            for layer4 in layer4_children:
                leaf_id = safe_id(layer3.get("id", ""), layer4.get("id", ""))
                leaf_name = " / ".join(
                    part for part in [str(layer3.get("name", "")).strip(), str(layer4.get("name", "")).strip()] if part
                )
                collapsed_first["children"].append(
                    {
                        "id": leaf_id,
                        "parent_id": collapsed_first["id"],
                        "depth": 2,
                        "name": leaf_name,
                        "kind": "Linked Semantic Gap",
                        "horizon": " / ".join(
                            part
                            for part in [
                                str(layer3.get("horizon", "")).strip(),
                                str(layer4.get("horizon", "")).strip(),
                            ]
                            if part
                        ),
                        "gap": linked_leaf_doc(layer3, layer4),
                        "document_text": linked_leaf_doc(layer3, layer4),
                        "source_layer3_id": layer3.get("id", ""),
                        "source_layer3_name": layer3.get("name", ""),
                        "source_layer4_id": layer4.get("id", ""),
                        "source_layer4_name": layer4.get("name", ""),
                    }
                )

        collapsed_root["children"].append(collapsed_first)

    return collapsed_root

def walk_tree(node: dict) -> Iterable[dict]:
    yield node
    for child in node.get("children", []) or []:
        yield from walk_tree(child)

def content_senses(entry: LemmaEntry) -> List[Sense]:
    return [sense for sense in entry.senses if sense.pos in CONTENT_POSES]

def load_candidates(
    *,
    wordnet_dir: Optional[Path],
    min_senses: int,
    single_word_only: bool,
    max_lemmas: Optional[int],
) -> List[LemmaEntry]:
    entries = load_wordnet(wordnet_dir)
    filtered: List[LemmaEntry] = []
    seen = set()
    for entry in entries:
        lemma = entry.lemma.lower()
        if lemma in seen:
            continue
        seen.add(lemma)
        if single_word_only and ("_" in lemma or "-" in lemma or " " in lemma):
            continue
        if lemma in STOPWORDS:
            continue
        if not tokenize(lemma):
            continue
        senses = content_senses(entry)
        if len(senses) > min_senses:
            filtered.append(LemmaEntry(lemma=lemma, senses=senses))

    filtered.sort(key=lambda item: (-len(item.senses), item.lemma))
    if max_lemmas:
        filtered = filtered[:max_lemmas]
    return filtered

def build_embeddings(
    leaves: Sequence[dict],
    candidates: Sequence[LemmaEntry],
    *,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    device: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    leaf_docs = {leaf["id"]: leaf.get("document_text") or field_text(leaf) for leaf in leaves}
    sense_docs = {
        sense.synset_id: sense.text
        for entry in candidates
        for sense in entry.senses
    }

    leaf_vecs = encode_with_sentence_transformers(
        list(leaf_docs),
        [leaf_docs[key] for key in leaf_docs],
        model_name=model_name,
        cache_dir=cache_dir,
        cache_label="future_routing_linked_leaf_nodes",
        batch_size=batch_size,
        device=device,
    )
    sense_vecs = encode_with_sentence_transformers(
        list(sense_docs),
        [sense_docs[key] for key in sense_docs],
        model_name=model_name,
        cache_dir=cache_dir,
        cache_label="future_routing_candidate_senses",
        batch_size=batch_size,
        device=device,
    )
    return leaf_vecs, sense_vecs

def leaf_options_for_one(
    entry: LemmaEntry,
    first_nodes: Sequence[dict],
    child_lookup: Dict[str, Sequence[dict]],
    leaf_vecs: Dict[str, np.ndarray],
    sense_vecs: Dict[str, np.ndarray],
    *,
    top_m: int,
) -> List[LeafOption]:
    options = []
    for first in first_nodes:
        first_scored: List[Tuple[float, dict, Sense]] = []
        best_by_leaf: Dict[str, Tuple[float, dict, Sense]] = {}
        for leaf in child_lookup[first["id"]]:
            leaf_vec = leaf_vecs[leaf["id"]]
            leaf_best_score = -1.0
            leaf_best_sense = entry.senses[0]
            for sense in entry.senses:
                score = float(cosine(sense_vecs[sense.synset_id], leaf_vec))
                first_scored.append((score, leaf, sense))
                if (score, sense.synset_id) > (leaf_best_score, leaf_best_sense.synset_id):
                    leaf_best_score = score
                    leaf_best_sense = sense
            best_by_leaf[leaf["id"]] = (leaf_best_score, leaf, leaf_best_sense)

        first_scored.sort(key=lambda item: (-item[0], item[1]["id"], item[2].synset_id))
        top_scores = [score for score, _, _ in first_scored[: max(1, top_m)]]
        gate_score = float(sum(top_scores) / len(top_scores))
        for leaf_score, leaf, sense in best_by_leaf.values():
            options.append(
                LeafOption(
                    lemma=entry.lemma,
                    sense_count=len(entry.senses),
                    gate_score=gate_score,
                    leaf_score=float(leaf_score),
                    first_id=first["id"],
                    first_name=first.get("name", ""),
                    leaf_id=leaf["id"],
                    leaf_name=leaf.get("name", ""),
                    nearest_sense_id=sense.synset_id,
                    nearest_sense_gloss=sense.gloss,
                )
            )
    return options

def build_leaf_options(
    candidates: Sequence[LemmaEntry],
    first_nodes: Sequence[dict],
    child_lookup: Dict[str, Sequence[dict]],
    leaf_vecs: Dict[str, np.ndarray],
    sense_vecs: Dict[str, np.ndarray],
    *,
    top_m: int,
) -> List[LeafOption]:
    options: List[LeafOption] = []
    for idx, entry in enumerate(candidates, start=1):
        if idx == 1 or idx % 100 == 0:
            print(f"[{idx}/{len(candidates)}] scoring leaf options for {entry.lemma}", flush=True)
        options.extend(
            leaf_options_for_one(
                entry,
                first_nodes,
                child_lookup,
                leaf_vecs,
                sense_vecs,
                top_m=top_m,
            )
        )
    options.sort(key=lambda item: (-item.gate_score, -item.leaf_score, item.first_id, item.leaf_id, item.lemma))
    return options

def collect_leaf_queues_from_options(
    options: Sequence[LeafOption],
    leaves: Sequence[dict],
    *,
    min_per_leaf: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int, bool]:
    leaf_queues: Dict[str, List[Dict[str, Any]]] = {leaf["id"]: [] for leaf in leaves}
    leaf_meta = {leaf["id"]: leaf for leaf in leaves}
    used_lemmas = set()

    def complete() -> bool:
        return all(len(items) >= min_per_leaf for items in leaf_queues.values())

    consumed = 0
    for global_rank, option in enumerate(options, start=1):
        if option.lemma in used_lemmas:
            continue
        queue = leaf_queues[option.leaf_id]
        if len(queue) >= min_per_leaf:
            continue
        leaf = leaf_meta[option.leaf_id]
        queue.append(
            {
                "leaf_order": len(queue) + 1,
                "global_rank": global_rank,
                "lemma": option.lemma,
                "sense_count": option.sense_count,
                "gate_score": option.gate_score,
                "leaf_score": option.leaf_score,
                "nearest_sense_id": option.nearest_sense_id,
                "nearest_sense_gloss": option.nearest_sense_gloss,
                "first_id": option.first_id,
                "first_name": option.first_name,
                "leaf_id": option.leaf_id,
                "leaf_name": option.leaf_name,
                "source_layer3_id": leaf.get("source_layer3_id", ""),
                "source_layer3_name": leaf.get("source_layer3_name", ""),
                "source_layer4_id": leaf.get("source_layer4_id", ""),
                "source_layer4_name": leaf.get("source_layer4_name", ""),
            }
        )
        used_lemmas.add(option.lemma)
        consumed = global_rank
        if complete():
            return leaf_queues, consumed, True
    return leaf_queues, consumed, complete()

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def write_leaf_csv(path: Path, leaf_queues: Dict[str, List[Dict[str, Any]]]) -> None:
    rows = [item for items in leaf_queues.values() for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-json", type=Path, default=DEFAULT_TREE_JSON)
    parser.add_argument("--wordnet-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-senses", type=int, default=3, help="Use lemmas with more than this many WordNet senses.")
    parser.add_argument("--min-per-leaf", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--single-word-only", action="store_true", help="Only use WordNet lemmas without spaces, hyphens or underscores.")
    parser.add_argument("--max-lemmas", type=int, default=None)
    parser.add_argument("--model-name", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source_tree = read_json(args.tree_json)
    collapsed_tree = collapse_future_tree(source_tree)
    first_nodes = collapsed_tree["children"]
    leaves = [child for first in first_nodes for child in first.get("children", []) or []]
    child_lookup = {first["id"]: first.get("children", []) or [] for first in first_nodes}

    print(f"Collapsed tree: {len(first_nodes)} first-domain nodes, {len(leaves)} linked leaves.", flush=True)
    candidates = load_candidates(
        wordnet_dir=args.wordnet_dir,
        min_senses=args.min_senses,
        single_word_only=args.single_word_only,
        max_lemmas=args.max_lemmas,
    )
    print(f"Loaded {len(candidates)} WordNet candidate lemmas with > {args.min_senses} senses.", flush=True)
    if not candidates:
        raise RuntimeError("No candidates were loaded.")

    leaf_vecs, sense_vecs = build_embeddings(
        leaves,
        candidates,
        model_name=args.model_name,
        cache_dir=args.embedding_cache_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    options = build_leaf_options(
        candidates,
        first_nodes,
        child_lookup,
        leaf_vecs,
        sense_vecs,
        top_m=args.top_m,
    )
    leaf_queues, consumed_rank, is_complete = collect_leaf_queues_from_options(
        options,
        leaves,
        min_per_leaf=args.min_per_leaf,
    )

    leaf_counts = {leaf_id: len(items) for leaf_id, items in leaf_queues.items()}
    sorted_counts = sorted(leaf_counts.items(), key=lambda item: (item[1], item[0]))
    metadata = {
        "source_tree_json": str(args.tree_json),
        "collapsed_rule": "root unchanged; first-domain nodes unchanged; each leaf links one original third-layer node with one original fourth-layer node.",
        "routing_strategy": "child_top5avg_plain with first-come-first-served leaf quotas",
        "routing_strategy_detail": "For each first-domain node, average the top 5 similarities between all candidate WordNet senses and all linked leaves under that first-domain. Then rank lemma-leaf options globally by first-domain gate score and leaf score, assigning each lemma once to the first leaf queue that still needs words.",
        "min_senses_exclusive": args.min_senses,
        "candidate_count": len(candidates),
        "leaf_option_count": len(options),
        "first_domain_count": len(first_nodes),
        "leaf_count": len(leaves),
        "min_per_leaf": args.min_per_leaf,
        "top_m": args.top_m,
        "consumed_global_rank": consumed_rank,
        "complete": is_complete,
        "model_name": args.model_name,
        "device": args.device,
    }
    payload = {
        "metadata": metadata,
        "leaf_counts": leaf_counts,
        "least_filled_leaves": [
            {
                "leaf_id": leaf_id,
                "count": count,
                "leaf_name": next((leaf.get("name", "") for leaf in leaves if leaf["id"] == leaf_id), ""),
            }
            for leaf_id, count in sorted_counts[:20]
        ],
        "leaf_queues": leaf_queues,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "future_tree_three_layer_linked.json", collapsed_tree)
    write_json(args.output_dir / "future_leaf_word_queues.json", payload)
    write_leaf_csv(args.output_dir / "future_leaf_word_queues.csv", leaf_queues)
    write_json(args.output_dir / "future_leaf_collection_summary.json", {"metadata": metadata, "leaf_counts": leaf_counts})

    print(json.dumps({"complete": is_complete, "consumed_global_rank": consumed_rank, "least_filled": sorted_counts[:10]}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_dir / 'future_tree_three_layer_linked.json'}")
    print(f"Wrote {args.output_dir / 'future_leaf_word_queues.json'}")
    print(f"Wrote {args.output_dir / 'future_leaf_word_queues.csv'}")
    print(f"Wrote {args.output_dir / 'future_leaf_collection_summary.json'}")

    if not is_complete and not args.allow_incomplete:
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
