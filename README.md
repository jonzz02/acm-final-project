# Bundestag Plenary Protocol Analysis Pipeline

## Overview

Automated pipeline for collecting, parsing, and analyzing German Bundestag plenary protocols from the 21st legislative period (2025). The system performs semantic clustering, polarization analysis, and network analysis of parliamentary discourse.

## Methodology

### 1. Data Collection

- **Source**: Bundestag DIP API (Plenarprotokolle endpoint)
- **Scope**: 21st legislative period, 2025 (date range: 2025-01-01 to 2025-12-31)
- **Method**: Cursor-based pagination for complete, reproducible retrieval
- **Artifacts**:
  - Protocol metadata (JSON) → `data/meta/`
  - XML transcripts (primary source) → `data/xml/`
  - PDF transcripts (fallback) → `data/pdf/`
  - Linked Vorgänge documents → `data/docs/`
  - Manifest file → `data/meta/manifest_wp21_2025.json`

### 2. Data Validation

Automated audit phase validates:
- File counts and completeness across directories
- Protocol metadata structure and document types
- File size distributions (detecting incomplete downloads)
- Temporal coverage and protocols-per-month distribution

### 3. Corpus Construction

**XML Parsing**:
- Session metadata extraction (date, number, legislative period)
- Agenda reconstruction from table of contents (TOP markers, titles, descriptions)
- Speech-level parsing with stable identifiers
- Speaker metadata: name, title, faction, role
- Speech-to-agenda linkage via XML cross-references

**Speaker Categorization**:
- **Parliament**: Speeches with faction affiliation
- **Government**: Chancellor, Ministers, State Secretaries
- **Presidency**: Bundestag President and Vice Presidents
- Enables focused analysis on parliamentary party discourse

**Outputs**: `data/results/speeches_parsed.csv`, `data/results/tops_with_descriptions.json`

### 4. Text Processing

**Quality Filtering**:
- Minimum speech length: 2,000 characters
- Excludes brief interjections and procedural remarks

**Text Cleaning** (for embeddings):
- Normalize dash variants and remove group prefixes
- Strip speaker headers and stage directions
- Remove formal salutations and closing boilerplate
- Normalize whitespace
- Output: `data/results/speeches_cleaned.csv`

**Linguistic Normalization** (for lexical baselines):
- Lemmatization with spaCy German model
- Stopword removal (standard + parliamentary custom list)
- Filter punctuation, numbers, short tokens
- Output: `data/results/speeches_normalized_bow.csv`

### 5. Feature Representation

**Lexical Space (TF–IDF)**:
- Unigrams and bigrams
- Document frequency thresholds to remove noise
- Letter-only tokens (preserves German umlauts and ß)
- Artifacts: `data/models/tfidf_*.{npz,pkl,txt,json}`

**Semantic Space (Embeddings)**:
- OpenAI embeddings on cleaned text
- Dense vectors for geometry-based analysis
- Artifacts: `data/embeddings/openai_embeddings.npy`

### 6. Topic Discovery

**Semantic Clustering**:
1. UMAP dimensionality reduction (50D, cosine metric)
2. HDBSCAN density clustering
3. Cluster characterization via top keywords
4. LLM-generated German labels (2–5 words per cluster)
5. Noise speeches assigned cluster_id = -1

**Lexical Baseline (NMF)**:
- Non-negative matrix factorization on TF–IDF
- Component count matches semantic cluster count
- Agreement measured via Adjusted Rand Index and Normalized Mutual Information

### 7. Polarization Analysis

**Topic-Level Polarization**:
- Focus: Parliamentary speeches only
- Party centroids computed per topic (cluster or meta-cluster)
- Polarization score: Mean pairwise cosine distance between party centroids
- Identifies most distant (conflict) and closest (consensus) party pairs

**Temporal Dynamics**:
- Monthly party centroids and polarization scores
- Tracks evolution of party divergence within topics over time

### 8. Network Analysis

**Party Similarity Network**:
- Edge weights: Topic-conditioned similarity aggregated across clusters
- Weighting: Minimum speech count per party pair per cluster (prevents sparse dominance)
- Metrics:
  - Weighted degree (strength)
  - Eigenvector centrality (on similarity weights)
  - Betweenness and closeness centrality (on distance = 1 - similarity)
  - Community detection (greedy modularity)

**Outputs**: 
- Similarity matrix CSV
- Edge list CSV
- Centrality measures CSV
- Network summary JSON

## Data Structure

```
data/
├── meta/               # Protocol metadata JSON files
├── xml/                # XML transcripts (primary source)
├── pdf/                # PDF transcripts (fallback)
├── docs/               # Linked Vorgänge documents
├── embeddings/         # OpenAI embeddings and metadata
├── models/             # TF-IDF models and party centroids
└── results/            # Parsed speeches, cleaned corpus, analytics
```

## Key Parameters

- **Speech length threshold**: 2,000 characters
- **UMAP dimensions**: 50
- **Clustering algorithm**: HDBSCAN (density-based)
- **TF-IDF n-grams**: 1–2
- **Embedding model**: OpenAI
- **Distance metric**: Cosine

## Reproducibility

- Cursor-based API pagination with manifest persistence
- Fixed random seeds for UMAP and clustering
- Cached LLM labels
- All intermediate artifacts saved
- Structured output directories

## Requirements

- Python 3.8+
- Dependencies: `pandas`, `numpy`, `scikit-learn`, `umap-learn`, `hdbscan`, `spacy`, `openai`, `networkx`
- German language model: `python -m spacy download de_core_news_sm`

## Analysis Outputs

1. **Exploratory**: Speech counts, length distributions, temporal patterns
2. **Topics**: Cluster labels, keywords, NMF baseline agreement
3. **Polarization**: Topic rankings, conflict/consensus pairs, monthly trends
4. **Network**: Party similarity graph, centrality measures, community structure

---

*Pipeline designed for transparency, reproducibility, and methodological rigor in computational political science.*

