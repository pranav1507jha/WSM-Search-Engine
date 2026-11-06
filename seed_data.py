# seed_data.py
import os
import json

DOCUMENTS = [
    {
        "filename": "doc01_intro_ir.txt",
        "title": "Introduction to Information Retrieval and Web Search",
        "author": "Manning and Raghavan",
        "category": "Foundations",
        "year": "2024",
        "language": "English",
        "acl": "public",
        "body": (
            "Information retrieval deals with the representation, storage, organization of, and access to "
            "unstructured text documents. Unlike structured relational databases that evaluate exact table "
            "predicates, web search engines process free text queries over billions of documents. Classic examples "
            "include searching Shakespeare plays for Brutus and Caesar but not Calpurnia using incidence matrices "
            "and inverted indexes."
        )
    },
    {
        "filename": "doc02_inverted_index.txt",
        "title": "Inverted Index Construction and Skip Pointers",
        "author": "Christopher Manning",
        "category": "Indexing",
        "year": "2024",
        "language": "English",
        "acl": "public",
        "body": (
            "An inverted index maps every distinct dictionary term to a sorted postings list of document identifiers "
            "and term frequencies. During boolean AND query processing, two postings lists are intersected in linear "
            "time using a simultaneous merge walk. Augmenting postings lists with evenly spaced skip pointers of length "
            "square root of L allows the intersection algorithm to bypass non-matching document IDs rapidly."
        )
    },
    {
        "filename": "doc03_boolean_retrieval.txt",
        "title": "Boolean Retrieval Model and Query Optimization",
        "author": "Hinrich Schutze",
        "category": "Retrieval Models",
        "year": "2023",
        "language": "English",
        "acl": "public",
        "body": (
            "The Boolean retrieval model evaluates queries connected by AND, OR, and NOT operators. Legal search "
            "systems like WestLaw rely on exact boolean matching and proximity operators such as limit within three "
            "words of statute. Query optimization processes conjunctions in order of increasing document frequency "
            "so the smallest intermediate postings set is intersected first."
        )
    },
    {
        "filename": "doc04_tokenization_morphology.txt",
        "title": "Tokenization Case Folding Stemming and Lemmatization",
        "author": "Martin Porter",
        "category": "Linguistics",
        "year": "2023",
        "language": "English",
        "acl": "public",
        "body": (
            "Linguistic preprocessing converts raw character streams into normalized index tokens. Equivalence class "
            "normalization removes periods in acronyms like U.S.A. and strips diacritics from words like resume. "
            "Porter stemming applies five sequential phases of suffix stripping rules to reduce automate, automatic, "
            "and automation to automat, while lemmatization uses morphological analysis to map inflectional forms "
            "like am, are, and is to their dictionary headword be."
        )
    },
    {
        "filename": "doc05_phrase_positional.txt",
        "title": "Phrase Queries Biword and Positional Indexes",
        "author": "Williams and Zobel",
        "category": "Indexing",
        "year": "2025",
        "language": "English",
        "acl": "public",
        "body": (
            "About ten percent of web search queries are phrase queries enclosed in quotation marks, such as "
            "stanford university palo alto. Biword indexes store every consecutive word pair as a vocabulary entry "
            "for immediate two-word lookup. Positional indexes record exact word offsets within each document, "
            "enabling arbitrary multi-word phrase verification and directional or windowed proximity search."
        )
    },
    {
        "filename": "doc06_wildcards_permuterm.txt",
        "title": "Wildcard Queries Permuterm and K-Gram Indexes",
        "author": "Prabhakar Raghavan",
        "category": "Tolerant Retrieval",
        "year": "2025",
        "language": "English",
        "acl": "public",
        "body": (
            "Wildcard queries support prefix, suffix, and infix pattern matching when users are uncertain of exact "
            "spelling. The permuterm index appends a terminal dollar symbol to each term and indexes all cyclic "
            "rotations so any single wildcard query translates into a prefix range lookup. Alternatively, a bigram "
            "or trigram index decomposes terms into overlapping character sequences and intersects candidate terms "
            "followed by regex post-filtering."
        )
    },
    {
        "filename": "doc07_spelling_correction.txt",
        "title": "Spelling Correction Edit Distance and Jaccard Overlap",
        "author": "Karen Kukich",
        "category": "Tolerant Retrieval",
        "year": "2024",
        "language": "English",
        "acl": "public",
        "body": (
            "Spelling correction repairs user typos in isolated query terms and multi-word phrases. Candidate terms "
            "are pruned rapidly by computing the Jaccard coefficient over character trigrams before running dynamic "
            "programming Levenshtein edit distance. Weighted edit distance incorporates QWERTY keyboard adjacency "
            "matrices, while context-sensitive correction inspects surrounding biword frequencies to fix real-word "
            "errors like flew form Heathrow."
        )
    },
    {
        "filename": "doc08_soundex_phonetic.txt",
        "title": "Soundex Phonetic Hashing for Proper Nouns",
        "author": "Robert Russell",
        "category": "Tolerant Retrieval",
        "year": "2022",
        "language": "English",
        "acl": "public",
        "body": (
            "Phonetic retrieval resolves pronunciation equivalents across variant surname spellings such as "
            "Chebyshev and Tchebycheff, or Herman and Hermann, or Robert and Rupert. The Soundex algorithm retains "
            "the initial letter, maps consonant groups to digits one through six, deduplicates adjacent identical "
            "digits, drops vowels, and pads the result to a four-character alphanumeric code."
        )
    },
    {
        "filename": "doc09_bsbi_mapreduce.txt",
        "title": "Scalable Index Construction BSBI and MapReduce",
        "author": "Jeffrey Dean",
        "category": "Architecture",
        "year": "2025",
        "language": "English",
        "acl": "public",
        "body": (
            "Constructing an inverted index over massive corpora exceeds main memory capacity and requires external "
            "sorting algorithms. Blocked Sort-Based Indexing segments term-document records into fixed memory "
            "blocks, quicksorts each block in RAM, and performs a multi-way disk merge. Distributed web indexers "
            "use MapReduce parser and inverter worker pools alongside dynamic auxiliary indexes and invalidation "
            "bit-vectors for real-time updates."
        )
    },
    {
        "filename": "doc10_dictionary_compression.txt",
        "title": "Heaps Law and Dictionary Compression Techniques",
        "author": "Alistair Moffat",
        "category": "Compression",
        "year": "2024",
        "language": "English",
        "acl": "public",
        "body": (
            "Heaps law predicts vocabulary growth as a power function of collection tokens. Fixed-width dictionary "
            "arrays waste substantial RAM because English words average eight characters. Storing the dictionary as "
            "a single concatenated string with block pointers every four terms and front-coding shared alphabetical "
            "prefixes like automata, automate, automatic, and automation reduces memory footprint dramatically."
        )
    },
    {
        "filename": "doc11_postings_compression.txt",
        "title": "Zipfs Law Variable Byte and Elias Gamma Postings Compression",
        "author": "Peter Elias",
        "category": "Compression",
        "year": "2024",
        "language": "English",
        "acl": "public",
        "body": (
            "Zipfs law states that term frequency is inversely proportional to term rank. Frequent words produce "
            "small gaps between consecutive document IDs in postings lists. Variable-byte compression encodes gaps "
            "using seven data bits and one continuation bit per byte for fast hardware alignment, whereas Elias "
            "gamma coding pairs a unary length prefix with a binary offset for bit-level entropy compression."
        )
    },
    {
        "filename": "doc12_zones_tfidf.txt",
        "title": "Parametric Search Zone Scoring and TF-IDF Vector Space",
        "author": "Gerard Salton",
        "category": "Retrieval Models",
        "year": "2025",
        "language": "English",
        "acl": "public",
        "body": (
            "Documents contain structured metadata fields and free-text zones such as title, author, and body. "
            "Linear zone combinations assign weighted scores to matches in each region. For free-text ranking, "
            "the vector space model combines sublinear logarithmic term frequency with inverse document frequency "
            "and computes cosine similarity between length-normalized query and document vectors."
        )
    },
    {
        "filename": "doc13_relevance_feedback.txt",
        "title": "Rocchio Relevance Feedback and Probabilistic Re-Ranking",
        "author": "Joseph Rocchio",
        "category": "Re-Ranking",
        "year": "2025",
        "language": "English",
        "acl": "public",
        "body": (
            "Local relevance feedback refines initial search rankings using explicit or implicit user judgments. "
            "The Rocchio algorithm shifts the query vector toward the centroid of relevant documents and away from "
            "non-relevant documents. Probabilistic relevance feedback estimates term weights via the Binary "
            "Independence Model, while pseudo-relevance feedback automatically expands queries using the top three "
            "retrieved results."
        )
    },
    {
        "filename": "doc14_thesaurus_querylogs.txt",
        "title": "Global Query Expansion Thesauri and Query Log Mining",
        "author": "Susan Dumais",
        "category": "Re-Ranking",
        "year": "2023",
        "language": "English",
        "acl": "public",
        "body": (
            "Global query expansion improves recall by augmenting queries with domain synonyms from a controlled "
            "thesaurus, mapping automobile to car and retrieval to search. Search engines also mine historical "
            "query logs and clickstreams to recommend co-occurring query terms and apply indirect relevance "
            "feedback boosts to frequently clicked documents."
        )
    },
    {
        "filename": "doc15_enterprise_security_acl.txt",
        "title": "Enterprise Search Access Control Lists and Restricted Roster",
        "author": "Security Administrator",
        "category": "Architecture",
        "year": "2026",
        "language": "English",
        "acl": "admin",
        "body": (
            "Enterprise document repositories enforce strict Access Control Lists so confidential records like the "
            "executive salary roster and internal audit reports are only returned to authenticated admin users. "
            "User group memberships are intersected with document ACL bit-vectors at query evaluation time."
        )
    }
]

THESAURUS = {
    "search": ["retrieval", "query", "lookup", "find"],
    "retrieval": ["search", "lookup", "recovery", "fetching"],
    "query": ["search", "question", "request", "expression"],
    "car": ["automobile", "vehicle", "motorcar"],
    "automobile": ["car", "vehicle"],
    "index": ["catalog", "dictionary", "lexicon", "postings"],
    "dictionary": ["lexicon", "vocabulary", "glossary"],
    "lexicon": ["dictionary", "vocabulary"],
    "compression": ["encoding", "compaction", "shrinkage"],
    "encoding": ["compression", "coding"],
    "fast": ["rapid", "quick", "accelerated"],
    "error": ["typo", "misspelling", "mistake"],
    "typo": ["error", "misspelling"],
    "ranking": ["scoring", "ordering", "grading", "reranking"],
    "scoring": ["ranking", "weighting"],
    "feedback": ["rocchio", "reranking", "refinement"],
    "stemming": ["lemmatization", "morphology", "normalization"],
    "lemmatization": ["stemming", "morphology"],
    "wildcard": ["permuterm", "kgram", "trigram", "pattern"],
    "soundex": ["phonetic", "homophone"]
}

SEED_QUERY_LOG = [
    "inverted index construction",
    "inverted index skip pointers",
    "boolean retrieval query optimization",
    "spelling correction edit distance",
    "spelling correction jaccard trigram",
    "dictionary compression front coding",
    "postings compression variable byte elias gamma",
    "relevance feedback rocchio algorithm",
    "vector space model cosine similarity tfidf",
    "wildcard queries permuterm index",
    "phrase queries biword positional index",
    "soundex phonetic matching algorithm",
    "porter stemming lemmatization normalization"
]

SEED_CLICKS = {
    "rocchio": {"13": 5, "12": 2},
    "compression": {"10": 4, "11": 6},
    "inverted index": {"2": 7, "1": 3},
    "spelling": {"7": 5, "6": 2}
}


def initialize_data_files(base_dir="."):
    docs_dir = os.path.join(base_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)

    for doc in DOCUMENTS:
        path = os.path.join(docs_dir, doc["filename"])
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Title: {doc['title']}\n")
                f.write(f"Author: {doc['author']}\n")
                f.write(f"Category: {doc['category']}\n")
                f.write(f"Year: {doc['year']}\n")
                f.write(f"Language: {doc['language']}\n")
                f.write(f"ACL: {doc['acl']}\n")
                f.write("---\n")
                f.write(doc["body"] + "\n")

    thesaurus_path = os.path.join(base_dir, "thesaurus.json")
    if not os.path.exists(thesaurus_path):
        with open(thesaurus_path, "w", encoding="utf-8") as f:
            json.dump(THESAURUS, f, indent=2)

    qlog_path = os.path.join(base_dir, "query_log.json")
    if not os.path.exists(qlog_path):
        with open(qlog_path, "w", encoding="utf-8") as f:
            json.dump(SEED_QUERY_LOG, f, indent=2)

    clicks_path = os.path.join(base_dir, "clicks.json")
    if not os.path.exists(clicks_path):
        with open(clicks_path, "w", encoding="utf-8") as f:
            json.dump(SEED_CLICKS, f, indent=2)


if __name__ == "__main__":
    initialize_data_files()
    print("Corpus documents, thesaurus.json, query_log.json, and clicks.json initialized.")