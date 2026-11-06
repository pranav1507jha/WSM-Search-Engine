# Web Search & Mining (WSM): Tolerant Boolean & Vector Space Retrieval Engine

**Developed by:** 
* Pranav Kumar Jha (B124086)
* Sai Swarup Panda (B124106)

🌐 **Live Deployed Website:** [https://wsm-search-engine-jzcp.onrender.com/](https://wsm-search-engine-jzcp.onrender.com/)

---

## 🚀 Overview
This project is a comprehensive Information Retrieval (IR) and Web Search system built from scratch in Python (Flask) and React (Tailwind CSS) without external search libraries[cite: 267, 268]. It implements foundational web search mining algorithms taught in the WSM curriculum, covering linguistic normalization, scalable indexing, tolerant retrieval, index compression, zone scoring, TF-IDF vector space ranking, and relevance feedback[cite: 267, 269, 270, 271].

---

## 🔍 Key Features Implemented

1. **Linguistic Preprocessing & Morphology:**
   - Tokenization, case-folding, and diacritic normalization[cite: 56, 57, 59, 269].
   - **Lemmatization** (headword mapping) and **Porter's 5-Phase Stemmer** / **Lovins Stemmer**[cite: 63, 65, 67].

2. **Scalable & Dynamic Indexing:**
   - Positional Inverted Index with $\sqrt{L}$ Skip Pointers for fast linear merge intersections[cite: 71, 74, 81, 269, 270].
   - Biword Index (2-word phrases) and Positional Index ($3+$ word phrases and `/k` proximity search)[cite: 77, 81, 84, 270].
   - **Dynamic Indexing:** Auxiliary index support for live document insertions and Invalidation Bit-Vectors for soft deletions[cite: 162].
   - **Enterprise Security:** Access Control Lists (ACL) restricting administrative documents by user roles[cite: 170].

3. **Boolean Retrieval & Optimization:**
   - Precedence parser supporting `AND`, `OR`, `NOT`, brackets `()`, and quotation marks `""`[cite: 269].
   - **Document Frequency ($df$) Query Optimization:** Evaluates smallest postings lists first[cite: 28, 269].

4. **Tolerant Retrieval Subsystem:**
   - **Wildcard Queries:** Permuterm Index ($$) and Bigram/Trigram K-Gram Index with regex post-filtering[cite: 95, 97].
   - **Spell Correction:** 3-Gram Jaccard overlap pruning + Levenshtein & QWERTY-Weighted Edit Distance dynamic programming + Context-Sensitive biword phrase correction[cite: 108, 111, 114, 116, 120].
   - **Soundex:** 6-step phonetic hashing for proper noun variations[cite: 130, 131].
   - **Global Expansion:** Manual synonym thesaurus and query log co-occurrence mining[cite: 125, 271].

5. **Index Compression Benchmarks:**
   - **Dictionary-as-a-String** with $k=4$ Blocking and Front-Coding[cite: 209, 211, 217, 270].
   - **Variable-Byte (VB)** and **Elias $\gamma$ (Gamma)** gap encoding for postings lists[cite: 186, 199, 270].

6. **Zone Scoring, TF-IDF & Re-Ranking:**
   - Linear zone weighting (`Title`, `Author`, `Body`) combined with sublinear TF-IDF Cosine Similarity[cite: 237, 258, 262].
   - **Re-Ranking & Relevance Feedback:** Explicit Rocchio Feedback, Probabilistic Binary Independence Model (BIM), Pseudo-Relevance Feedback, and Indirect Clickstream Re-Ranking[cite: 271].

---

## 🛠️ Tech Stack
* **Backend:** Python, Flask, Gunicorn[cite: 268]
* **Frontend:** React (Vite / CDN), Tailwind CSS[cite: 268]
* **Hosting:** Render[cite: 276]