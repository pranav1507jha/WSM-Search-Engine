# engine.py
import os
import re
import math
import json
import unicodedata
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

from index_compression import (
    BlockedDictionary,
    VariableByteCodec,
    EliasGammaCodec,
    compute_compression_report
)


# ==============================================================================
# 1. LINGUISTIC PREPROCESSING (Tokenization, Stemming, Lemmatization, Soundex)
# ==============================================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"
}

LEMMA_DICTIONARY = {
    "am": "be", "are": "be", "is": "be", "was": "be", "were": "be", "been": "be", "being": "be",
    "has": "have", "had": "have", "having": "have",
    "does": "do", "did": "do", "done": "do",
    "cars": "car", "car's": "car", "automobiles": "automobile",
    "children": "child", "mice": "mouse", "feet": "foot", "teeth": "tooth", "men": "man", "women": "woman",
    "plays": "play", "played": "play", "playing": "play",
    "queries": "query", "queried": "query", "querying": "query",
    "indexes": "index", "indices": "index", "indexed": "index", "indexing": "index",
    "postings": "posting", "documents": "document", "terms": "term",
    "matrices": "matrix", "vertices": "vertex", "analyses": "analysis",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "flew": "fly", "flown": "fly", "flying": "fly",
    "ran": "run", "running": "run", "runs": "run",
    "wrote": "write", "written": "write", "writing": "write",
    "built": "build", "building": "build", "builds": "build",
    "found": "find", "finding": "find", "finds": "find",
    "colors": "color", "colours": "color", "colour": "color"
}


class PorterStemmer:
    """Implements Porter's 5-phase suffix stripping algorithm for English."""
    @staticmethod
    def _is_consonant(word: str, i: int) -> bool:
        ch = word[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return True if i == 0 else not PorterStemmer._is_consonant(word, i - 1)
        return True

    @classmethod
    def _measure(cls, stem: str) -> int:
        """Computes consonant-vowel sequence measure m in [C](VC)^m[V]."""
        n = len(stem)
        i = 0
        while i < n and cls._is_consonant(stem, i):
            i += 1
        m = 0
        while i < n:
            while i < n and not cls._is_consonant(stem, i):
                i += 1
            if i >= n:
                break
            m += 1
            while i < n and cls._is_consonant(stem, i):
                i += 1
        return m

    @classmethod
    def _contains_vowel(cls, stem: str) -> bool:
        return any(not cls._is_consonant(stem, i) for i in range(len(stem)))

    @classmethod
    def stem(cls, word: str) -> str:
        if len(word) <= 2:
            return word
        w = word.lower()

        # Phase 1a: Plurals
        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif not w.endswith("ss") and w.endswith("s"):
            w = w[:-1]

        # Phase 1b: Past participles (-eed, -ed, -ing)
        if w.endswith("eed"):
            stem = w[:-3]
            if cls._measure(stem) > 0:
                w = w[:-1]
        elif w.endswith("ed"):
            stem = w[:-2]
            if cls._contains_vowel(stem):
                w = stem
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
        elif w.endswith("ing"):
            stem = w[:-3]
            if cls._contains_vowel(stem):
                w = stem
                if w.endswith(("at", "bl", "iz")):
                    w += "e"

        # Phase 1c: y -> i
        if w.endswith("y") and cls._contains_vowel(w[:-1]):
            w = w[:-1] + "i"

        # Phase 2: Derivational suffixes
        step2_rules = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
            ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")
        ]
        for suffix, replacement in step2_rules:
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                if cls._measure(stem) > 0:
                    w = stem + replacement
                break

        # Phase 3
        step3_rules = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"),
            ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")
        ]
        for suffix, replacement in step3_rules:
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                if cls._measure(stem) > 0:
                    w = stem + replacement
                break

        # Phase 4: (m > 1) suffix removal
        step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        ]
        for suffix in step4_suffixes:
            if w.endswith(suffix):
                stem = w[:-len(suffix)]
                if cls._measure(stem) > 1:
                    if suffix == "ion":
                        if stem.endswith(("s", "t")):
                            w = stem
                    else:
                        w = stem
                break

        # Phase 5a: Final -e removal
        if w.endswith("e"):
            stem = w[:-1]
            if cls._measure(stem) > 1:
                w = stem

        return w


class LovinsStemmer:
    """Single-pass longest-match suffix removal stemmer (Lovins approach)."""
    SUFFIXES = sorted([
        "ational", "fulness", "iveness", "ization", "ousness", "biliti",
        "ation", "alism", "aliti", "iviti", "ement", "ence", "ance", "able",
        "ible", "ment", "ness", "ally", "ious", "less", "ings", "ing", "ies",
        "ion", "ate", "ive", "ous", "ize", "ful", "ism", "ist", "ed", "er", "es", "ly", "s"
    ], key=len, reverse=True)

    @classmethod
    def stem(cls, word: str) -> str:
        w = word.lower()
        if len(w) <= 3:
            return w
        for suf in cls.SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[:-len(suf)]
        return w


class LinguisticProcessor:
    """Handles normalization, tokenization, stop-word filtering, lemmatization, stemming, and Soundex."""

    @staticmethod
    def normalize_text(text: str) -> str:
        # Remove diacritics (e.g., résumé -> resume, Tübingen -> Tubingen)
        nfkd = unicodedata.normalize("NFKD", text)
        ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
        # Normalize dotted acronyms (e.g., U.S.A. -> USA)
        ascii_text = re.sub(r"\b(?:[A-Za-z]\.){2,}", lambda m: m.group(0).replace(".", ""), ascii_text)
        # Remove possessive apostrophes (Finland's -> Finland)
        ascii_text = re.sub(r"'s\b", "", ascii_text)
        ascii_text = ascii_text.replace("'", "")
        # Split hyphens into spaces so Hewlett-Packard yields hewlett and packard
        ascii_text = ascii_text.replace("-", " ")
        return ascii_text.lower()

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        norm = cls.normalize_text(text)
        return re.findall(r"[a-z0-9]+", norm)

    @classmethod
    def lemmatize(cls, token: str) -> str:
        t = token.lower()
        if t in LEMMA_DICTIONARY:
            return LEMMA_DICTIONARY[t]
        if len(t) > 4 and t.endswith("ies"):
            return t[:-3] + "y"
        if len(t) > 4 and t.endswith("es") and not t.endswith(("aes", "ees", "oes")):
            if t.endswith(("ches", "shes", "xes", "sses", "zes")):
                return t[:-2]
            return t[:-1]
        if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            return t[:-1]
        if len(t) > 5 and t.endswith("ing"):
            base = t[:-3]
            return base
        if len(t) > 4 and t.endswith("ed"):
            return t[:-2]
        return t

    @classmethod
    def process_token(
        cls,
        token: str,
        use_stemming: bool = False,
        stemmer_type: str = "porter",
        use_lemmatization: bool = False
    ) -> str:
        t = token.lower()
        if use_lemmatization:
            t = cls.lemmatize(t)
        if use_stemming:
            if stemmer_type == "lovins":
                t = LovinsStemmer.stem(t)
            else:
                t = PorterStemmer.stem(t)
        return t

    @staticmethod
    def soundex(word: str) -> str:
        """
        6-Step Soundex Algorithm:
        1. Retain first letter (uppercase).
        2. Replace A, E, I, O, U, H, W, Y with '0'.
        3. Map B,F,P,V->1; C,G,J,K,Q,S,X,Z->2; D,T->3; L->4; M,N->5; R->6.
        4. Collapse consecutive identical digits.
        5. Remove all '0's.
        6. Pad with trailing zeros and truncate to 4 characters.
        """
        clean = re.sub(r"[^a-zA-Z]", "", word).upper()
        if not clean:
            return "0000"

        first_letter = clean[0]
        mapping = {
            "B": "1", "F": "1", "P": "1", "V": "1",
            "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
            "D": "3", "T": "3",
            "L": "4",
            "M": "5", "N": "5",
            "R": "6"
        }

        digits = []
        for ch in clean:
            digits.append(mapping.get(ch, "0"))

        # Deduplicate adjacent equal digits
        collapsed = [digits[0]]
        for d in digits[1:]:
            if d != collapsed[-1]:
                collapsed.append(d)

        # Drop first position and remove zeros
        filtered = [d for d in collapsed[1:] if d != "0"]
        code = first_letter + "".join(filtered)
        return (code + "000")[:4]


# ==============================================================================
# 2. TOLERANT RETRIEVAL (Permuterm, K-Gram, Jaccard + Edit Distance Spell Check)
# ==============================================================================

QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk", "i": "uojkl", "o": "ipkl", "p": "ol",
    "a": "qwsxz", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn", "j": "yuihknm", "k": "uiojlm", "l": "opk",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk"
}


class TolerantIndex:
    """Maintains Permuterm Index, Bigram/Trigram Indexes, Soundex Index, and Spell Checker."""
    def __init__(self):
        self.vocabulary: Set[str] = set()
        self.doc_freqs: Dict[str, int] = {}
        self.permuterm_entries: List[Tuple[str, str]] = []  # Sorted (rotated_key, original_term)
        self.bigram_index: Dict[str, Set[str]] = defaultdict(set)
        self.trigram_index: Dict[str, Set[str]] = defaultdict(set)
        self.soundex_index: Dict[str, Set[str]] = defaultdict(set)
        self.biword_counts: Dict[str, int] = defaultdict(int)

    def build(self, vocab_df: Dict[str, int], biword_doc_counts: Dict[str, int]):
        self.vocabulary = set(vocab_df.keys())
        self.doc_freqs = dict(vocab_df)
        self.biword_counts = dict(biword_doc_counts)
        self.permuterm_entries.clear()
        self.bigram_index.clear()
        self.trigram_index.clear()
        self.soundex_index.clear()

        for term in self.vocabulary:
            # 1. Permuterm rotations with '$'
            padded = term + "$"
            for i in range(len(padded)):
                rotated = padded[i:] + padded[:i]
                self.permuterm_entries.append((rotated, term))

            # 2. Bigrams & Trigrams with '$' boundary symbol
            bounded = f"${term}$"
            for i in range(len(bounded) - 1):
                bg = bounded[i: i + 2]
                self.bigram_index[bg].add(term)
            for i in range(len(bounded) - 2):
                tg = bounded[i: i + 3]
                self.trigram_index[tg].add(term)

            # 3. Soundex
            scode = LinguisticProcessor.soundex(term)
            self.soundex_index[scode].add(term)

        self.permuterm_entries.sort(key=lambda x: x[0])

    def resolve_wildcard_permuterm(self, pattern: str) -> Tuple[List[str], str]:
        """
        Rotates wildcard query so '*' appears at the right end, then scans sorted Permuterm lexicon.
        Supports X*, *X, X*Y, *X*, and multi-wildcard X*Y*Z via post-filtering.
        """
        pat = pattern.lower()
        parts = pat.split("*")
        if len(parts) == 2:
            prefix, suffix = parts[0], parts[1]
            lookup_prefix = f"{suffix}${prefix}"
        else:
            # For multi-wildcard like X*Y*Z, rotate longest literal segment with '$'
            non_empty = [p for p in parts if p]
            lookup_prefix = non_empty[0] if non_empty else ""

        regex_pat = "^" + ".*".join(re.escape(p) for p in parts) + "$"
        regex = re.compile(regex_pat)

        matched = set()
        for rot_key, orig_term in self.permuterm_entries:
            if rot_key.startswith(lookup_prefix):
                if regex.match(orig_term):
                    matched.add(orig_term)

        explanation = f"Permuterm lookup prefix '{lookup_prefix}*' -> matched {len(matched)} term(s)"
        return sorted(matched), explanation

    def resolve_wildcard_kgram(self, pattern: str) -> Tuple[List[str], str]:
        """Resolves wildcard pattern using Bigram/Trigram index intersection + regex post-filter."""
        pat = pattern.lower()
        bounded = f"${pat}$"
        segments = [seg for seg in bounded.split("*") if len(seg) >= 2]

        grams = []
        for seg in segments:
            for i in range(len(seg) - 1):
                grams.append(seg[i: i + 2])

        if grams:
            candidates = set(self.bigram_index.get(grams[0], set()))
            for g in grams[1:]:
                candidates &= self.bigram_index.get(g, set())
        else:
            candidates = set(self.vocabulary)

        regex_pat = "^" + ".*".join(re.escape(p) for p in pat.split("*")) + "$"
        regex = re.compile(regex_pat)
        filtered = sorted(t for t in candidates if regex.match(t))
        explanation = f"K-Gram intersection on {grams} ({len(candidates)} pre-filter -> {len(filtered)} post-filter)"
        return filtered, explanation

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Standard Dynamic Programming Edit Distance (Insert, Delete, Replace = 1)."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j - 1] + cost,  # Replace / Match
                    dp[i - 1][j] + 1,         # Delete
                    dp[i][j - 1] + 1          # Insert
                )
        return dp[m][n]

    @staticmethod
    def weighted_edit_distance(s1: str, s2: str) -> float:
        """QWERTY-aware Weighted Edit Distance (adjacent keyboard substitutions cost 0.5)."""
        m, n = len(s1), len(s2)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = float(i)
        for j in range(n + 1):
            dp[0][j] = float(j)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                c1, c2 = s1[i - 1], s2[j - 1]
                if c1 == c2:
                    sub_cost = 0.0
                elif c2 in QWERTY_NEIGHBORS.get(c1, ""):
                    sub_cost = 0.5
                else:
                    sub_cost = 1.0
                dp[i][j] = min(
                    dp[i - 1][j - 1] + sub_cost,
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0
                )
        return round(dp[m][n], 2)

    @staticmethod
    def trigram_set(word: str) -> Set[str]:
        bounded = f"${word.lower()}$"
        if len(bounded) < 3:
            return {bounded}
        return {bounded[i: i + 3] for i in range(len(bounded) - 2)}

    def suggest_isolated_spelling(self, word: str, max_edit_dist: int = 2, top_k: int = 3) -> List[dict]:
        """
        1. Prunes vocabulary using 3-Gram Jaccard Overlap (|X ∩ Y| / |X ∪ Y|).
        2. Evaluates Levenshtein and QWERTY-Weighted Edit Distance on candidates.
        3. Ranks by lowest Weighted Edit Distance, highest Jaccard, and highest Document Frequency.
        """
        q = word.lower()
        if q in self.vocabulary or len(q) <= 2:
            return []

        q_trigrams = self.trigram_set(q)
        candidate_terms = set()
        for tg in q_trigrams:
            candidate_terms |= self.trigram_index.get(tg, set())

        if not candidate_terms:
            candidate_terms = set(self.vocabulary)

        scored = []
        for cand in candidate_terms:
            if abs(len(cand) - len(q)) > max_edit_dist:
                continue
            c_trigrams = self.trigram_set(cand)
            union_len = len(q_trigrams | c_trigrams)
            jaccard = len(q_trigrams & c_trigrams) / union_len if union_len else 0.0

            edit_d = self.levenshtein_distance(q, cand)
            if edit_d <= max_edit_dist:
                w_edit_d = self.weighted_edit_distance(q, cand)
                df = self.doc_freqs.get(cand, 1)
                scored.append({
                    "term": cand,
                    "edit_distance": edit_d,
                    "weighted_edit_distance": w_edit_d,
                    "jaccard": round(jaccard, 3),
                    "df": df
                })

        scored.sort(key=lambda x: (x["weighted_edit_distance"], -x["jaccard"], -x["df"]))
        return scored[:top_k]

    def suggest_context_sensitive_phrase(self, tokens: List[str]) -> Optional[str]:
        """
        Detects real-word or isolated typos in multi-word queries (e.g., 'flew form heathrow' -> 'flew from heathrow')
        by checking biword corpus frequencies of single-term edit-distance-1 variations.
        """
        if len(tokens) < 2:
            return None

        best_phrase = None
        best_score = 0

        # Current biword score
        current_score = sum(
            self.biword_counts.get(f"{tokens[i]} {tokens[i + 1]}", 0)
            for i in range(len(tokens) - 1)
        )
        if current_score > 0 and all(t in self.vocabulary for t in tokens):
            return None

        for idx, tok in enumerate(tokens):
            # Gather candidates within edit distance <= 2
            candidates = [
                c for c in self.vocabulary
                if abs(len(c) - len(tok)) <= 2 and 0 < self.levenshtein_distance(tok, c) <= 2
            ]
            for cand in candidates:
                trial = list(tokens)
                trial[idx] = cand
                biword_hits = sum(
                    self.biword_counts.get(f"{trial[i]} {trial[i + 1]}", 0) * 10
                    for i in range(len(trial) - 1)
                )
                df_Bonus = sum(self.doc_freqs.get(t, 0) for t in trial)
                total = biword_hits + df_Bonus
                if biword_hits > 0 and total > best_score:
                    best_score = total
                    best_phrase = " ".join(trial)

        return best_phrase


# ==============================================================================
# 3. SKIP-POINTER POSTINGS LISTS & LINEAR MERGE
# ==============================================================================

class SkipPostingsList:
    """
    Stores a sorted postings list augmented with evenly spaced sqrt(L) skip pointers.
    """
    def __init__(self, doc_ids: List[int]):
        self.doc_ids = sorted(set(doc_ids))
        self.length = len(self.doc_ids)
        self.skip_step = int(math.floor(math.sqrt(self.length))) if self.length >= 4 else 0
        self.skips: Dict[int, int] = {}
        if self.skip_step >= 2:
            for i in range(0, self.length - self.skip_step, self.skip_step):
                self.skips[i] = i + self.skip_step

    def has_skip(self, idx: int) -> bool:
        return idx in self.skips

    def skip_target(self, idx: int) -> int:
        return self.skips[idx]


def intersect_with_skips(p1: SkipPostingsList, p2: SkipPostingsList) -> Tuple[List[int], int]:
    """
    Intersects two SkipPostingsLists using skip pointers.
    Returns (matched_doc_ids, num_comparisons_saved_by_skips).
    """
    answer = []
    i, j = 0, 0
    skips_taken = 0
    a, b = p1.doc_ids, p2.doc_ids

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            answer.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            if p1.has_skip(i) and a[p1.skip_target(i)] <= b[j]:
                while p1.has_skip(i) and a[p1.skip_target(i)] <= b[j]:
                    i = p1.skip_target(i)
                    skips_taken += 1
            else:
                i += 1
        else:
            if p2.has_skip(j) and b[p2.skip_target(j)] <= a[i]:
                while p2.has_skip(j) and b[p2.skip_target(j)] <= a[i]:
                    j = p2.skip_target(j)
                    skips_taken += 1
            else:
                j += 1
    return answer, skips_taken


# ==============================================================================
# 4. MASTER SEARCH ENGINE (BSBI, Dynamic Index, Boolean, Zones, TF-IDF, Feedback)
# ==============================================================================

class SearchEngine:
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.docs_dir = os.path.join(base_dir, "documents")
        self.thesaurus_path = os.path.join(base_dir, "thesaurus.json")
        self.qlog_path = os.path.join(base_dir, "query_log.json")
        self.clicks_path = os.path.join(base_dir, "clicks.json")

        # Document Store & Dynamic Indexing State
        self.documents: Dict[int, dict] = {}
        self.auxiliary_doc_ids: Set[int] = set()
        self.deleted_doc_ids: Set[int] = set()  # Invalidation Bit-Vector

        # Indexes
        self.positional_index: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.skip_postings: Dict[str, SkipPostingsList] = {}
        self.biword_index: Dict[str, Set[int]] = defaultdict(set)
        self.zone_index: Dict[str, Dict[str, Set[int]]] = {
            "title": defaultdict(set),
            "author": defaultdict(set),
            "body": defaultdict(set)
        }
        self.parametric_index: Dict[str, Dict[str, Set[int]]] = {
            "category": defaultdict(set),
            "year": defaultdict(set),
            "language": defaultdict(set)
        }

        # Compression & Tolerant structures
        self.sorted_terms: List[str] = []
        self.blocked_dict: Optional[BlockedDictionary] = None
        self.compressed_postings_vb: Dict[str, bytes] = {}
        self.tolerant = TolerantIndex()

        # Global Expansion & Logs
        self.thesaurus: Dict[str, List[str]] = {}
        self.query_log: List[str] = []
        self.clicks: Dict[str, Dict[str, int]] = {}
        self.bsbi_stats: dict = {}

        self.load_all()

    def load_all(self):
        self._load_json_stores()
        self._load_documents_from_disk()
        self.rebuild_indexes()

    def _load_json_stores(self):
        if os.path.exists(self.thesaurus_path):
            with open(self.thesaurus_path, "r", encoding="utf-8") as f:
                self.thesaurus = json.load(f)
        if os.path.exists(self.qlog_path):
            with open(self.qlog_path, "r", encoding="utf-8") as f:
                self.query_log = json.load(f)
        if os.path.exists(self.clicks_path):
            with open(self.clicks_path, "r", encoding="utf-8") as f:
                self.clicks = json.load(f)

    def _parse_doc_file(self, doc_id: int, filename: str, raw_text: str) -> dict:
        meta = {
            "id": doc_id,
            "filename": filename,
            "title": filename,
            "author": "Unknown",
            "category": "General",
            "year": "2025",
            "language": "English",
            "acl": "public",
            "body": raw_text
        }
        if "---\n" in raw_text:
            header_part, body_part = raw_text.split("---\n", 1)
            meta["body"] = body_part.strip()
            for line in header_part.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k in meta:
                        meta[k] = v
        else:
            lines = raw_text.strip().splitlines()
            if lines:
                meta["title"] = lines[0].strip()
                meta["body"] = "\n".join(lines[1:]).strip()
        return meta

    def _load_documents_from_disk(self):
        self.documents.clear()
        if not os.path.exists(self.docs_dir):
            return
        files = sorted(f for f in os.listdir(self.docs_dir) if f.endswith(".txt"))
        for idx, fname in enumerate(files, start=1):
            fpath = os.path.join(self.docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                raw = f.read()
            self.documents[idx] = self._parse_doc_file(idx, fname, raw)

    def rebuild_indexes(
        self,
        use_stemming: bool = False,
        stemmer_type: str = "porter",
        use_lemmatization: bool = False,
        remove_stopwords: bool = False
    ):
        """
        Constructs all inverted, positional, biword, zone, permuterm, k-gram, Soundex,
        and compressed structures using a simulated BSBI block-sort-merge pipeline.
        """
        self.positional_index.clear()
        self.skip_postings.clear()
        self.biword_index.clear()
        for z in self.zone_index.values():
            z.clear()
        for p in self.parametric_index.values():
            p.clear()

        bsbi_records = []  # (term, doc_id, position)
        biword_doc_counts: Dict[str, int] = defaultdict(int)

        active_docs = {
            did: doc for did, doc in self.documents.items()
            if did not in self.deleted_doc_ids
        }

        for doc_id, doc in active_docs.items():
            # Parametric metadata indexing
            self.parametric_index["category"][doc["category"].lower()].add(doc_id)
            self.parametric_index["year"][str(doc["year"]).lower()].add(doc_id)
            self.parametric_index["language"][doc["language"].lower()].add(doc_id)

            # Zone indexing (title, author, body)
            for zone_name in ("title", "author", "body"):
                raw_zone_tokens = LinguisticProcessor.tokenize(doc.get(zone_name, ""))
                for tok in raw_zone_tokens:
                    if remove_stopwords and tok in STOP_WORDS:
                        continue
                    norm_tok = LinguisticProcessor.process_token(
                        tok, use_stemming, stemmer_type, use_lemmatization
                    )
                    self.zone_index[zone_name][norm_tok].add(doc_id)

            # Full document stream (Title + Body) for Positional and Biword indexes
            full_text = f"{doc['title']} {doc['body']}"
            raw_tokens = LinguisticProcessor.tokenize(full_text)
            processed_tokens = []
            for tok in raw_tokens:
                if remove_stopwords and tok in STOP_WORDS:
                    continue
                norm_tok = LinguisticProcessor.process_token(
                    tok, use_stemming, stemmer_type, use_lemmatization
                )
                processed_tokens.append(norm_tok)

            for pos, term in enumerate(processed_tokens):
                bsbi_records.append((term, doc_id, pos))

            # Biword index construction
            seen_biwords_in_doc = set()
            for i in range(len(processed_tokens) - 1):
                bw = f"{processed_tokens[i]} {processed_tokens[i + 1]}"
                self.biword_index[bw].add(doc_id)
                seen_biwords_in_doc.add(bw)
            for bw in seen_biwords_in_doc:
                biword_doc_counts[bw] += 1

        # Simulate BSBI Block Quicksort + Multi-way Merge
        block_size = 250
        num_blocks = max(1, math.ceil(len(bsbi_records) / block_size))
        sorted_runs = []
        for b_idx in range(num_blocks):
            block = bsbi_records[b_idx * block_size: (b_idx + 1) * block_size]
            block.sort(key=lambda r: (r[0], r[1], r[2]))
            sorted_runs.append(block)

        # Merge sorted runs into Positional Index
        merged_records = sorted(bsbi_records, key=lambda r: (r[0], r[1], r[2]))
        for term, doc_id, pos in merged_records:
            self.positional_index[term][doc_id].append(pos)

        self.bsbi_stats = {
            "total_token_occurrences": len(bsbi_records),
            "block_size_records": block_size,
            "num_sorted_runs": num_blocks,
            "merge_tree_layers": math.ceil(math.log2(num_blocks)) if num_blocks > 1 else 1
        }

        # Build Skip-Pointer Postings and Compressed Structures
        self.sorted_terms = sorted(self.positional_index.keys())
        postings_map: Dict[str, List[int]] = {}
        vocab_df: Dict[str, int] = {}

        for term in self.sorted_terms:
            doc_ids = sorted(self.positional_index[term].keys())
            postings_map[term] = doc_ids
            vocab_df[term] = len(doc_ids)
            self.skip_postings[term] = SkipPostingsList(doc_ids)
            self.compressed_postings_vb[term] = VariableByteCodec.encode(doc_ids)

        self.blocked_dict = BlockedDictionary(self.sorted_terms, block_size=4)
        self.tolerant.build(vocab_df, biword_doc_counts)

    # --------------------------------------------------------------------------
    # Dynamic Indexing Operations (Auxiliary Index + Invalidation Bit-Vector)
    # --------------------------------------------------------------------------

    def add_document_dynamically(self, title: str, author: str, category: str, year: str, acl: str, body: str) -> dict:
        new_id = max(self.documents.keys(), default=0) + 1
        fname = f"doc{new_id:02d}_dynamic.txt"
        doc_obj = {
            "id": new_id,
            "filename": fname,
            "title": title.strip() or f"Untitled Document {new_id}",
            "author": author.strip() or "Anonymous",
            "category": category.strip() or "General",
            "year": str(year).strip() or "2026",
            "language": "English",
            "acl": acl.strip().lower() or "public",
            "body": body.strip()
        }
        self.documents[new_id] = doc_obj
        self.auxiliary_doc_ids.add(new_id)
        self.rebuild_indexes()
        return doc_obj

    def delete_document_dynamically(self, doc_id: int) -> bool:
        if doc_id in self.documents:
            self.deleted_doc_ids.add(doc_id)
            self.rebuild_indexes()
            return True
        return False

    def restore_deleted_documents(self):
        self.deleted_doc_ids.clear()
        self.rebuild_indexes()

    def _allowed_docs_for_role(self, user_role: str) -> Set[int]:
        """Filters active documents by Invalidation Bit-Vector and ACL Role."""
        allowed = set()
        for did, doc in self.documents.items():
            if did in self.deleted_doc_ids:
                continue
            if doc.get("acl", "public") == "admin" and user_role != "admin":
                continue
            allowed.add(did)
        return allowed

    # --------------------------------------------------------------------------
    # Phrase & Proximity Lookup (Biword for 2 words, Positional for >=3 or /k)
    # --------------------------------------------------------------------------

    def evaluate_phrase(
        self,
        phrase_str: str,
        use_stemming: bool = False,
        stemmer_type: str = "porter",
        use_lemmatization: bool = False
    ) -> Tuple[Set[int], str]:
        raw_tokens = LinguisticProcessor.tokenize(phrase_str)
        tokens = [
            LinguisticProcessor.process_token(t, use_stemming, stemmer_type, use_lemmatization)
            for t in raw_tokens
        ]
        if not tokens:
            return set(), "Empty phrase"

        if len(tokens) == 1:
            t = tokens[0]
            docs = set(self.positional_index.get(t, {}).keys())
            return docs, f"Single-term lookup '{t}' -> {len(docs)} doc(s)"

        if len(tokens) == 2:
            bw = f"{tokens[0]} {tokens[1]}"
            docs = set(self.biword_index.get(bw, set()))
            return docs, f"Biword Index direct lookup for \"{bw}\" -> {len(docs)} doc(s)"

        # >= 3 words: Positional Index verification
        candidate_docs = set(self.positional_index.get(tokens[0], {}).keys())
        for t in tokens[1:]:
            candidate_docs &= set(self.positional_index.get(t, {}).keys())

        verified_docs = set()
        for did in candidate_docs:
            first_positions = self.positional_index[tokens[0]][did]
            for start_pos in first_positions:
                if all(
                    (start_pos + offset) in self.positional_index[tokens[offset]][did]
                    for offset in range(1, len(tokens))
                ):
                    verified_docs.add(did)
                    break

        return (
            verified_docs,
            f"Positional Index verification for \"{ ' '.join(tokens) }\" -> {len(verified_docs)} doc(s)"
        )

    def evaluate_proximity(
        self,
        term1: str,
        term2: str,
        k_window: int,
        use_stemming: bool = False,
        stemmer_type: str = "porter",
        use_lemmatization: bool = False
    ) -> Tuple[Set[int], str]:
        """Evaluates 'term1 /k term2' using Positional Index merge within k words."""
        t1 = LinguisticProcessor.process_token(term1, use_stemming, stemmer_type, use_lemmatization)
        t2 = LinguisticProcessor.process_token(term2, use_stemming, stemmer_type, use_lemmatization)

        docs1 = self.positional_index.get(t1, {})
        docs2 = self.positional_index.get(t2, {})
        common_docs = set(docs1.keys()) & set(docs2.keys())

        matched_docs = set()
        for did in common_docs:
            pos_list1 = docs1[did]
            pos_list2 = docs2[did]
            for p1 in pos_list1:
                if any(abs(p1 - p2) <= k_window for p2 in pos_list2):
                    matched_docs.add(did)
                    break

        return (
            matched_docs,
            f"Positional proximity '{t1} /{k_window} {t2}' (within {k_window} words) -> {len(matched_docs)} doc(s)"
        )

    # --------------------------------------------------------------------------
    # Boolean Query Parser & Frequency-Based Query Optimizer
    # --------------------------------------------------------------------------

    def execute_boolean_query(
        self,
        query: str,
        options: dict
    ) -> dict:
        use_stemming = options.get("use_stemming", False)
        stemmer_type = options.get("stemmer_type", "porter")
        use_lemmatization = options.get("use_lemmatization", False)
        remove_stopwords = options.get("remove_stopwords", False)
        wildcard_mode = options.get("wildcard_mode", "permuterm")
        user_role = options.get("user_role", "public")

        # Rebuild if morphological flags changed
        self.rebuild_indexes(use_stemming, stemmer_type, use_lemmatization, remove_stopwords)
        universe = self._allowed_docs_for_role(user_role)
        trace_steps = []

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Tokenize Boolean expression preserving quotes, /k proximity, parentheses, AND, OR, NOT
        token_pattern = re.compile(
            r'"[^"]+"'
            r'|[A-Za-z0-9*]+\s+/\d+\s+[A-Za-z0-9*]+'
            r'|\(|\)'
            r'|\bAND\b|\bOR\b|\bNOT\b'
            r'|[^\s()"]+',
            re.IGNORECASE
        )
        raw_tokens = token_pattern.findall(query.strip())

        # Normalize operators and insert implicit AND between adjacent operands
        normalized_tokens = []
        positive_query_terms = []

        def is_operand(tok: str) -> bool:
            u = tok.upper()
            return u not in ("AND", "OR", "NOT", "(", ")")

        for tok in raw_tokens:
            u = tok.upper()
            if u in ("AND", "OR", "NOT", "(", ")"):
                if u in ("(", "NOT") and normalized_tokens:
                    prev = normalized_tokens[-1]
                    if is_operand(prev) or prev == ")":
                        normalized_tokens.append("AND")
                normalized_tokens.append(u)
            else:
                if normalized_tokens:
                    prev = normalized_tokens[-1]
                    if is_operand(prev) or prev == ")":
                        normalized_tokens.append("AND")
                normalized_tokens.append(tok)

        # Validate parentheses and operator placement
        depth = 0
        for idx, tok in enumerate(normalized_tokens):
            if tok == "(":
                depth += 1
            elif tok == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("Unbalanced closing parenthesis ')' in query.")
            elif tok in ("AND", "OR"):
                if idx == 0 or idx == len(normalized_tokens) - 1:
                    raise ValueError(f"Dangling boolean operator '{tok}'.")
                if normalized_tokens[idx + 1] in ("AND", "OR", ")"):
                    raise ValueError(f"Invalid operator sequence near '{tok}'.")
            elif tok == "NOT":
                if idx == len(normalized_tokens) - 1 or normalized_tokens[idx + 1] in ("AND", "OR", ")"):
                    raise ValueError("Dangling 'NOT' operator without operand.")
        if depth != 0:
            raise ValueError("Missing closing parenthesis ')' in query.")

        # Convert to Postfix (Reverse Polish Notation) via Shunting-Yard
        precedence = {"NOT": 3, "AND": 2, "OR": 1}
        output_queue = []
        op_stack = []

        for tok in normalized_tokens:
            if tok == "(":
                op_stack.append(tok)
            elif tok == ")":
                while op_stack and op_stack[-1] != "(":
                    output_queue.append(op_stack.pop())
                op_stack.pop()
            elif tok in precedence:
                while (
                    op_stack
                    and op_stack[-1] in precedence
                    and (
                        precedence[op_stack[-1]] > precedence[tok]
                        or (precedence[op_stack[-1]] == precedence[tok] and tok != "NOT")
                    )
                ):
                    output_queue.append(op_stack.pop())
                op_stack.append(tok)
            else:
                output_queue.append(tok)

        while op_stack:
            output_queue.append(op_stack.pop())

        # Helper to resolve any operand token to a document ID set
        def evaluate_atom(atom: str) -> Set[int]:
            # 1. Phrase query in quotes
            if atom.startswith('"') and atom.endswith('"'):
                inner = atom[1:-1]
                docs, step_msg = self.evaluate_phrase(inner, use_stemming, stemmer_type, use_lemmatization)
                trace_steps.append(step_msg)
                for w in LinguisticProcessor.tokenize(inner):
                    positive_query_terms.append(
                        LinguisticProcessor.process_token(w, use_stemming, stemmer_type, use_lemmatization)
                    )
                return docs & universe

            # 2. Proximity query: term1 /k term2
            prox_match = re.match(r"^([A-Za-z0-9*]+)\s+/(\d+)\s+([A-Za-z0-9*]+)$", atom)
            if prox_match:
                t1, k_str, t2 = prox_match.group(1), prox_match.group(2), prox_match.group(3)
                docs, step_msg = self.evaluate_proximity(t1, t2, int(k_str), use_stemming, stemmer_type, use_lemmatization)
                trace_steps.append(step_msg)
                positive_query_terms.extend([
                    LinguisticProcessor.process_token(t1, use_stemming, stemmer_type, use_lemmatization),
                    LinguisticProcessor.process_token(t2, use_stemming, stemmer_type, use_lemmatization)
                ])
                return docs & universe

            # 3. Wildcard query containing '*'
            if "*" in atom:
                if wildcard_mode == "kgram":
                    matched_terms, w_msg = self.tolerant.resolve_wildcard_kgram(atom)
                else:
                    matched_terms, w_msg = self.tolerant.resolve_wildcard_permuterm(atom)
                trace_steps.append(w_msg)
                union_docs = set()
                for mt in matched_terms:
                    union_docs |= set(self.positional_index.get(mt, {}).keys())
                    positive_query_terms.append(mt)
                return union_docs & universe

            # 4. Standard single term (verified via Compressed Blocked Dictionary & VB Decoding)
            clean_toks = LinguisticProcessor.tokenize(atom)
            if not clean_toks:
                return set()
            norm_t = LinguisticProcessor.process_token(clean_toks[0], use_stemming, stemmer_type, use_lemmatization)
            positive_query_terms.append(norm_t)

            term_id = self.blocked_dict.lookup(norm_t) if self.blocked_dict else None
            if term_id is not None and norm_t in self.compressed_postings_vb:
                decoded_docs = set(VariableByteCodec.decode(self.compressed_postings_vb[norm_t])) & universe
                trace_steps.append(
                    f"Term '{atom}' -> normalized '{norm_t}' (BlockedDict ID #{term_id}, df={len(decoded_docs)})"
                )
                return decoded_docs
            else:
                trace_steps.append(f"Term '{atom}' -> normalized '{norm_t}' (df=0, not in dictionary)")
                return set()

        # Evaluate RPN stack with skip-pointer intersection & df optimization logging
        eval_stack: List[Tuple[Set[int], str]] = []
        for item in output_queue:
            if item == "NOT":
                operand_docs, label = eval_stack.pop()
                comp = universe - operand_docs
                trace_steps.append(f"NOT ({label}) -> complement against {len(universe)} active docs = {len(comp)} doc(s)")
                eval_stack.append((comp, f"NOT {label}"))
            elif item == "AND":
                right_docs, r_label = eval_stack.pop()
                left_docs, l_label = eval_stack.pop()
                # Frequency optimization: intersect smaller set first
                if len(left_docs) <= len(right_docs):
                    first_set, second_set = left_docs, right_docs
                    order_msg = f"'{l_label}' (size={len(left_docs)}) AND '{r_label}' (size={len(right_docs)})"
                else:
                    first_set, second_set = right_docs, left_docs
                    order_msg = f"'{r_label}' (size={len(right_docs)}) AND '{l_label}' (size={len(left_docs)}) [Optimized by smaller df]"

                merged, skips_used = intersect_with_skips(
                    SkipPostingsList(list(first_set)),
                    SkipPostingsList(list(second_set))
                )
                res_set = set(merged)
                trace_steps.append(
                    f"Intersect {order_msg} -> {len(res_set)} match(es) (Skip pointers jumped {skips_used} step(s))"
                )
                eval_stack.append((res_set, f"({l_label} AND {r_label})"))
            elif item == "OR":
                right_docs, r_label = eval_stack.pop()
                left_docs, l_label = eval_stack.pop()
                union_set = left_docs | right_docs
                trace_steps.append(f"Union '{l_label}' OR '{r_label}' -> {len(union_set)} doc(s)")
                eval_stack.append((union_set, f"({l_label} OR {r_label})"))
            else:
                docs = evaluate_atom(item)
                eval_stack.append((docs, item))

        final_docs = eval_stack[0][0] if eval_stack else set()

        # Rank matching Boolean documents using TF-IDF + Zone score
        weights = {t: 1.0 for t in positive_query_terms if t}
        ranked_results = self._score_and_format_documents(
            candidate_docs=final_docs,
            query_weights=weights,
            zone_weights=options.get("zone_weights", {"title": 0.5, "author": 0.2, "body": 0.3}),
            raw_query=query
        )

        # Spell check suggestions if 0 results
        spell_suggestions = []
        for raw_w in LinguisticProcessor.tokenize(query):
            if raw_w.upper() not in ("AND", "OR", "NOT"):
                suggs = self.tolerant.suggest_isolated_spelling(raw_w)
                if suggs:
                    spell_suggestions.append({"original": raw_w, "candidates": suggs})

        return {
            "mode": "boolean",
            "query": query,
            "total_hits": len(ranked_results),
            "results": ranked_results,
            "trace": trace_steps,
            "spell_suggestions": spell_suggestions
        }

    # --------------------------------------------------------------------------
    # Advanced Ranked Retrieval (Tolerant Expansion + Zones + TF-IDF + Re-Ranking)
    # --------------------------------------------------------------------------

    def execute_advanced_query(self, query: str, options: dict) -> dict:
        use_stemming = options.get("use_stemming", False)
        stemmer_type = options.get("stemmer_type", "porter")
        use_lemmatization = options.get("use_lemmatization", False)
        remove_stopwords = options.get("remove_stopwords", False)
        wildcard_mode = options.get("wildcard_mode", "permuterm")
        user_role = options.get("user_role", "public")

        enable_spell = options.get("enable_spell", True)
        enable_soundex = options.get("enable_soundex", False)
        enable_thesaurus = options.get("enable_thesaurus", False)
        enable_qlog = options.get("enable_qlog", False)

        feedback_mode = options.get("feedback_mode", "none")  # none, rocchio, bim, pseudo, clicks
        relevant_ids = [int(x) for x in options.get("relevant_ids", [])]
        nonrelevant_ids = [int(x) for x in options.get("nonrelevant_ids", [])]
        zone_weights = options.get("zone_weights", {"title": 0.5, "author": 0.2, "body": 0.3})
        parametric_filters = options.get("parametric_filters", {})

        self.rebuild_indexes(use_stemming, stemmer_type, use_lemmatization, remove_stopwords)
        universe = self._allowed_docs_for_role(user_role)
        trace_steps = []
        expanded_chips = []
        spell_suggestions = []

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Log query
        clean_q_str = query.strip()
        if clean_q_str.lower() not in [q.lower() for q in self.query_log]:
            self.query_log.append(clean_q_str)
            with open(self.qlog_path, "w", encoding="utf-8") as f:
                json.dump(self.query_log, f, indent=2)

        # Apply Parametric Metadata Filters first
        filtered_universe = set(universe)
        for field_name in ("category", "year", "language"):
            f_val = parametric_filters.get(field_name, "all")
            if f_val and f_val.lower() != "all":
                matching_field_docs = self.parametric_index[field_name].get(f_val.lower(), set())
                filtered_universe &= matching_field_docs
                trace_steps.append(
                    f"Parametric Filter [{field_name.capitalize()} = {f_val}] -> {len(filtered_universe)} candidate doc(s)"
                )

        # Extract raw tokens and handle wildcards
        raw_words = re.findall(r"[a-zA-Z0-9*]+", clean_q_str.lower())
        query_weights: Dict[str, float] = defaultdict(float)

        for rw in raw_words:
            if remove_stopwords and rw in STOP_WORDS:
                trace_steps.append(f"Removed stop-word '{rw}'")
                continue

            if "*" in rw:
                if wildcard_mode == "kgram":
                    w_terms, w_msg = self.tolerant.resolve_wildcard_kgram(rw)
                else:
                    w_terms, w_msg = self.tolerant.resolve_wildcard_permuterm(rw)
                trace_steps.append(w_msg)
                for wt in w_terms[:6]:
                    query_weights[wt] = max(query_weights[wt], 0.85)
                    expanded_chips.append({"term": wt, "source": f"Wildcard ({rw})", "weight": 0.85})
                continue

            norm_t = LinguisticProcessor.process_token(rw, use_stemming, stemmer_type, use_lemmatization)
            query_weights[norm_t] = 1.0
            if norm_t != rw:
                trace_steps.append(f"Morphological normalization: '{rw}' -> '{norm_t}'")

            # 1. Isolated Spell Correction (if term not in dictionary or enable_spell active)
            if enable_spell and norm_t not in self.positional_index:
                suggs = self.tolerant.suggest_isolated_spelling(rw)
                if suggs:
                    spell_suggestions.append({"original": rw, "candidates": suggs})
                    top_corr = LinguisticProcessor.process_token(
                        suggs[0]["term"], use_stemming, stemmer_type, use_lemmatization
                    )
                    query_weights[top_corr] = max(query_weights[top_corr], 0.9)
                    expanded_chips.append({
                        "term": top_corr,
                        "source": f"Spell Fix ({rw}, d={suggs[0]['weighted_edit_distance']}, J={suggs[0]['jaccard']})",
                        "weight": 0.9
                    })
                    trace_steps.append(
                        f"Spell correction auto-expanded '{rw}' -> '{top_corr}' "
                        f"(Weighted Edit Dist={suggs[0]['weighted_edit_distance']}, Jaccard={suggs[0]['jaccard']})"
                    )

            # 2. Soundex Phonetic Expansion
            if enable_soundex:
                scode = LinguisticProcessor.soundex(rw)
                phonetic_matches = sorted(self.tolerant.soundex_index.get(scode, set()))
                trace_steps.append(f"Soundex('{rw}') = {scode} -> matches: {phonetic_matches}")
                weight_val = 1.0 if norm_t not in self.positional_index else 0.45
                for pm in phonetic_matches[:5]:
                    if pm != norm_t:
                        query_weights[pm] = max(query_weights[pm], weight_val)
                        expanded_chips.append({"term": pm, "source": f"Soundex ({scode})", "weight": weight_val})

            # 3. Manual Thesaurus Expansion
            if enable_thesaurus:
                syns = self.thesaurus.get(rw, []) + self.thesaurus.get(norm_t, [])
                for syn in syns:
                    norm_syn = LinguisticProcessor.process_token(syn, use_stemming, stemmer_type, use_lemmatization)
                    if norm_syn in self.positional_index and norm_syn not in query_weights:
                        query_weights[norm_syn] = 0.5
                        expanded_chips.append({"term": norm_syn, "source": f"Thesaurus ({rw})", "weight": 0.5})
                        trace_steps.append(f"Thesaurus synonym added: '{rw}' -> '{norm_syn}' (w=0.5)")

        # 4. Query Log Co-occurrence Expansion
        if enable_qlog:
            co_counts: Dict[str, int] = defaultdict(int)
            raw_set = set(raw_words)
            for past_q in self.query_log:
                past_toks = set(LinguisticProcessor.tokenize(past_q))
                if raw_set & past_toks:
                    for pt in past_toks - raw_set:
                        if pt not in STOP_WORDS:
                            co_counts[pt] += 1
            top_co = sorted(co_counts.items(), key=lambda x: -x[1])[:3]
            for co_term, _ in top_co:
                norm_co = LinguisticProcessor.process_token(co_term, use_stemming, stemmer_type, use_lemmatization)
                if norm_co in self.positional_index and norm_co not in query_weights:
                    query_weights[norm_co] = 0.35
                    expanded_chips.append({"term": norm_co, "source": "Query Log Co-occurrence", "weight": 0.35})
                    trace_steps.append(f"Query Log co-occurrence added: '{norm_co}' (w=0.35)")

        # Context-sensitive biword phrase suggestion
        did_you_mean = self.tolerant.suggest_context_sensitive_phrase(raw_words)

        # Initial Pass Scoring
        initial_results = self._score_and_format_documents(
            candidate_docs=filtered_universe,
            query_weights=query_weights,
            zone_weights=zone_weights,
            raw_query=clean_q_str
        )

        # ----------------------------------------------------------------------
        # 5. Re-Ranking & Relevance Feedback (Rocchio, BIM, Pseudo, Clicks)
        # ----------------------------------------------------------------------
        if feedback_mode == "rocchio" and (relevant_ids or nonrelevant_ids):
            query_weights, fb_chips, fb_msg = self._apply_rocchio_feedback(
                query_weights, relevant_ids, nonrelevant_ids
            )
            expanded_chips.extend(fb_chips)
            trace_steps.append(fb_msg)
            initial_results = self._score_and_format_documents(
                filtered_universe, query_weights, zone_weights, clean_q_str
            )

        elif feedback_mode == "bim" and relevant_ids:
            query_weights, fb_chips, fb_msg = self._apply_probabilistic_bim_feedback(
                query_weights, relevant_ids, len(filtered_universe)
            )
            expanded_chips.extend(fb_chips)
            trace_steps.append(fb_msg)
            initial_results = self._score_and_format_documents(
                filtered_universe, query_weights, zone_weights, clean_q_str
            )

        elif feedback_mode == "pseudo" and initial_results:
            top_pseudo_ids = [r["id"] for r in initial_results[:3]]
            query_weights, fb_chips, fb_msg = self._apply_rocchio_feedback(
                query_weights, top_pseudo_ids, [], label="Pseudo-Relevance (Top 3)"
            )
            expanded_chips.extend(fb_chips)
            trace_steps.append(fb_msg)
            initial_results = self._score_and_format_documents(
                filtered_universe, query_weights, zone_weights, clean_q_str
            )

        elif feedback_mode == "clicks":
            click_map = self.clicks.get(clean_q_str.lower(), {})
            if click_map:
                for res in initial_results:
                    c_count = click_map.get(str(res["id"]), 0)
                    if c_count > 0:
                        boost = min(0.25, 0.06 * c_count)
                        res["score"] = round(min(1.0, res["score"] + boost), 4)
                        res["click_boost"] = round(boost, 4)
                initial_results.sort(key=lambda r: -r["score"])
                trace_steps.append(f"Applied Indirect Clickstream Re-Ranking boost from {click_map}")

        return {
            "mode": "advanced",
            "query": query,
            "did_you_mean": did_you_mean,
            "total_hits": len(initial_results),
            "results": initial_results,
            "trace": trace_steps,
            "expanded_terms": expanded_chips,
            "spell_suggestions": spell_suggestions,
            "suggestions": self.get_query_suggestions(clean_q_str)
        }

    # --------------------------------------------------------------------------
    # TF-IDF Vector Space + Linear Zone Combination Scoring
    # --------------------------------------------------------------------------

    def _compute_doc_tfidf_vector(self, doc_id: int, total_docs: int) -> Dict[str, float]:
        vec: Dict[str, float] = {}
        for term, doc_map in self.positional_index.items():
            if doc_id in doc_map:
                tf = len(doc_map[doc_id])
                df = len(doc_map)
                wf = 1.0 + math.log10(tf) if tf > 0 else 0.0
                idf = math.log10((total_docs + 1) / (df + 0.5))
                vec[term] = wf * idf
        # L2 length normalize
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _score_and_format_documents(
        self,
        candidate_docs: Set[int],
        query_weights: Dict[str, float],
        zone_weights: dict,
        raw_query: str
    ) -> List[dict]:
        total_docs = max(1, len(self.documents) - len(self.deleted_doc_ids))
        w_title = float(zone_weights.get("title", 0.5))
        w_author = float(zone_weights.get("author", 0.2))
        w_body = float(zone_weights.get("body", 0.3))
        w_sum = (w_title + w_author + w_body) or 1.0
        w_title, w_author, w_body = w_title / w_sum, w_author / w_sum, w_body / w_sum

        # Construct normalized query TF-IDF vector
        q_vec: Dict[str, float] = {}
        for term, q_wt in query_weights.items():
            if term in self.positional_index:
                df = len(self.positional_index[term])
                idf = math.log10((total_docs + 1) / (df + 0.5))
                q_vec[term] = q_wt * idf
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {t: v / q_norm for t, v in q_vec.items()}

        results = []
        for did in candidate_docs:
            doc = self.documents[did]
            d_vec = self._compute_doc_tfidf_vector(did, total_docs)

            # Cosine similarity
            cosine_score = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)

            # Linear Zone Score
            q_terms = list(query_weights.keys())
            if q_terms:
                title_hit = 1.0 if any(did in self.zone_index["title"].get(t, set()) for t in q_terms) else 0.0
                author_hit = 1.0 if any(did in self.zone_index["author"].get(t, set()) for t in q_terms) else 0.0
                body_hit = 1.0 if any(did in self.zone_index["body"].get(t, set()) for t in q_terms) else 0.0
                zone_score = w_title * title_hit + w_author * author_hit + w_body * body_hit
            else:
                zone_score = 0.0

            if cosine_score <= 0.0 and zone_score <= 0.0:
                continue

            combined_score = round(0.7 * cosine_score + 0.3 * zone_score, 4)
            results.append({
                "id": did,
                "title": doc["title"],
                "author": doc["author"],
                "category": doc["category"],
                "year": doc["year"],
                "acl": doc["acl"],
                "is_auxiliary": did in self.auxiliary_doc_ids,
                "filename": doc["filename"],
                "score": combined_score,
                "cosine_score": round(cosine_score, 4),
                "zone_score": round(zone_score, 4),
                "snippet": self._build_snippet(doc["body"], list(query_weights.keys()) + LinguisticProcessor.tokenize(raw_query))
            })

        results.sort(key=lambda r: (-r["score"], r["id"]))
        return results

    @staticmethod
    def _build_snippet(body: str, highlight_terms: List[str]) -> str:
        words = body.split()
        snippet_text = " ".join(words[:36]) + ("..." if len(words) > 36 else "")
        return snippet_text

    # --------------------------------------------------------------------------
    # Rocchio & Probabilistic BIM Relevance Feedback
    # --------------------------------------------------------------------------

    def _apply_rocchio_feedback(
        self,
        base_weights: Dict[str, float],
        rel_ids: List[int],
        nonrel_ids: List[int],
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.15,
        label: str = "Rocchio Feedback"
    ) -> Tuple[Dict[str, float], List[dict], str]:
        total_docs = max(1, len(self.documents) - len(self.deleted_doc_ids))
        rel_Centroid: Dict[str, float] = defaultdict(float)
        nonrel_Centroid: Dict[str, float] = defaultdict(float)

        for did in rel_ids:
            if did in self.documents:
                d_vec = self._compute_doc_tfidf_vector(did, total_docs)
                for t, w in d_vec.items():
                    rel_Centroid[t] += w / len(rel_ids)

        for did in nonrel_ids:
            if did in self.documents:
                d_vec = self._compute_doc_tfidf_vector(did, total_docs)
                for t, w in d_vec.items():
                    nonrel_Centroid[t] += w / len(nonrel_ids)

        all_candidate_terms = set(base_weights.keys()) | set(rel_Centroid.keys())
        updated_scores: Dict[str, float] = {}
        for t in all_candidate_terms:
            if t in STOP_WORDS:
                continue
            val = alpha * base_weights.get(t, 0.0) + beta * rel_Centroid.get(t, 0.0) - gamma * nonrel_Centroid.get(t, 0.0)
            if val > 0.05:
                updated_scores[t] = round(val, 3)

        # Keep original terms + top 6 new expansion terms
        new_terms = [
            (t, s) for t, s in updated_scores.items()
            if t not in base_weights
        ]
        new_terms.sort(key=lambda x: -x[1])
        top_new = new_terms[:6]

        final_weights = {t: updated_scores.get(t, w) for t, w in base_weights.items()}
        chips = []
        for t, w in top_new:
            final_weights[t] = w
            chips.append({"term": t, "source": label, "weight": w})

        msg = (
            f"{label} applied (|Dr|={len(rel_ids)}, |Dnr|={len(nonrel_ids)}) -> "
            f"added {[t for t, _ in top_new]}"
        )
        return final_weights, chips, msg

    def _apply_probabilistic_bim_feedback(
        self,
        base_weights: Dict[str, float],
        rel_ids: List[int],
        N: int
    ) -> Tuple[Dict[str, float], List[dict], str]:
        R = len(rel_ids)
        candidate_scores = []
        for term, doc_map in self.positional_index.items():
            if term in STOP_WORDS or term in base_weights:
                continue
            df = len(doc_map)
            vr = sum(1 for did in rel_ids if did in doc_map)
            if vr == 0:
                continue
            p = (vr + 0.5) / (R + 1.0)
            u = (df - vr + 0.5) / max(1.0, (N - R + 1.0))
            p = min(max(p, 0.001), 0.999)
            u = min(max(u, 0.001), 0.999)
            c_weight = math.log10((p * (1.0 - u)) / (u * (1.0 - p)))
            if c_weight > 0:
                candidate_scores.append((term, round(c_weight, 3)))

        candidate_scores.sort(key=lambda x: -x[1])
        top_bim = candidate_scores[:6]

        final_weights = dict(base_weights)
        chips = []
        for t, w in top_bim:
            final_weights[t] = w
            chips.append({"term": t, "source": "Probabilistic BIM", "weight": w})

        msg = f"Probabilistic BIM Feedback (R={R}) -> added {[t for t, _ in top_bim]}"
        return final_weights, chips, msg

    # --------------------------------------------------------------------------
    # Click Recording, Suggestions & Index Stats
    # --------------------------------------------------------------------------

    def record_click(self, query: str, doc_id: int):
        q_key = query.strip().lower()
        if not q_key:
            return
        if q_key not in self.clicks:
            self.clicks[q_key] = {}
        d_key = str(doc_id)
        self.clicks[q_key][d_key] = self.clicks[q_key].get(d_key, 0) + 1
        with open(self.clicks_path, "w", encoding="utf-8") as f:
            json.dump(self.clicks, f, indent=2)

    def get_query_suggestions(self, partial_q: str) -> List[str]:
        p = partial_q.strip().lower()
        if not p:
            return self.query_log[:5]
        p_words = set(LinguisticProcessor.tokenize(p))
        matches = []
        for lq in self.query_log:
            if lq.lower() == p:
                continue
            if p in lq.lower() or (p_words & set(LinguisticProcessor.tokenize(lq))):
                matches.append(lq)
        return matches[:5]

    def get_stats(self) -> dict:
        postings_map = {
            t: sorted(self.positional_index[t].keys())
            for t in self.sorted_terms
        }
        comp_report = compute_compression_report(self.sorted_terms, postings_map)
        active_count = len(self.documents) - len(self.deleted_doc_ids)

        return {
            "corpus": {
                "total_documents": len(self.documents),
                "active_documents": active_count,
                "auxiliary_documents": len(self.auxiliary_doc_ids),
                "deleted_documents": len(self.deleted_doc_ids),
                "distinct_terms": len(self.sorted_terms),
                "distinct_biwords": len(self.biword_index),
                "permuterm_rotations": len(self.tolerant.permuterm_entries),
                "distinct_trigrams": len(self.tolerant.trigram_index),
                "soundex_buckets": len(self.tolerant.soundex_index)
            },
            "bsbi": self.bsbi_stats,
            "compression": comp_report
        }