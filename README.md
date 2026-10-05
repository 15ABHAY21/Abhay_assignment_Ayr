# Engineering RAG Engine

A multimodal Retrieval-Augmented Generation (RAG) system for extracting, indexing, retrieving, and reasoning over complex **engineering documents, patents, technical diagrams, and structured spreadsheets**.

The system combines **hierarchical document parsing, hybrid retrieval, Text-to-SQL, multimodal vision, and grounded generation** while maintaining document- and page-level provenance for every answer.

---

## 1. Architecture Overview

The engine uses an intent-based routing layer to determine whether a user query should be answered from:

* **Unstructured knowledge** → PDFs, patents, technical documents, diagrams
* **Structured knowledge** → CSV/XLSX engineering datasets

```text
                         ┌──────────────────┐
                         │    User Query    │
                         └────────┬─────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Query Intent Router │
                       └──────────┬──────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        ┌──────────────────┐              ┌──────────────────┐
        │ Unstructured RAG │              │ Structured SQL   │
        │      Pipeline    │              │     Pipeline     │
        └────────┬─────────┘              └────────┬─────────┘
                 │                                 │
        ┌────────▼─────────┐              ┌────────▼─────────┐
        │ Hybrid Retrieval │              │   Text-to-SQL    │
        │ Vector + FTS     │              │     DuckDB       │
        └────────┬─────────┘              └────────┬─────────┘
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Context Construction│
                       └──────────┬──────────┘
                                  ▼
                       ┌─────────────────────┐
                       │   LLM Generation    │
                       │ + Grounded Evidence │
                       └──────────┬──────────┘
                                  ▼
                       Answer + Document/Page
                              Citations
```

### Unstructured Pipeline

PDFs are processed using **semantic hierarchical chunking**. Each chunk retains:

* Document name
* Page number
* Section/heading context
* Extracted text
* Associated figures/tables where applicable

A **300-character sliding overlap** reduces information loss across chunk boundaries.

Retrieval uses **hybrid search**:

1. **Vector similarity** for semantic matching
2. **Tantivy full-text search** for exact keyword/alphanumeric matching

This is particularly useful for engineering identifiers such as:

`EV-BMS-100`, `IEC-62133`, sensor IDs, part numbers, error codes, and model numbers.

### Structured Pipeline

CSV and XLSX files are ingested into a local **DuckDB** database.

For structured questions:

```text
Natural Language Query
        ↓
   Text-to-SQL
        ↓
DuckDB Execution
        ↓
Result Rows → Column-Mapped Dictionaries
        ↓
LLM Answer Generation
```

The complete result rows are passed to the answering model so that relationships between multiple columns are preserved.

---

## 2. Document Processing & Multimodal Extraction

The extraction pipeline is designed to operate locally using open-source parsing components.

### Docling

**Docling** is used as the core document parser because engineering PDFs often contain much more than plain text.

It provides:

* Layout-aware document parsing
* Reading-order detection
* Table extraction
* Figure detection
* OCR integration
* Structured document representation

### Layout & Reading Order

Layout detection identifies elements such as:

* Titles
* Paragraphs
* Captions
* Tables
* Figures

Spatial relationships are then used to reconstruct the correct reading order, including multi-column technical documents.

### Table Extraction

**TableFormer** is used for visually understanding table structures, including:

* Rows and columns
* Cell boundaries
* Spanning cells

The extracted tables are converted into structured Markdown rather than being flattened into meaningless text.

### OCR

For scanned or image-only pages, **RapidOCR/PaddleOCR** is used to recover textual content from legacy documents and patents.

### Figure & Diagram Understanding

Detected engineering figures are cropped and passed to a **vision-language model**.

This enables extraction of:

* Reference numerals
* Labels
* Diagram text
* Component relationships
* Important visual context

As a result, information contained only inside diagrams can participate in retrieval and question answering.

---

## 3. Model & Library Choices

| Component           | Technology                 | Why                                                                |
| ------------------- | -------------------------- | ------------------------------------------------------------------ |
| Document Parser     | **Docling**                | Layout-aware parsing, tables, figures and OCR integration          |
| OCR                 | **RapidOCR / PaddleOCR**   | Local processing of scanned documents                              |
| Table Understanding | **TableFormer**            | Better preservation of complex table structures                    |
| Vector Store        | **LanceDB**                | Lightweight local vector database with efficient similarity search |
| Full-Text Search    | **Tantivy**                | Fast lexical search and exact engineering-code matching            |
| Structured DB       | **DuckDB**                 | Fast, embedded analytical SQL engine for CSV/XLSX                  |
| Embeddings          | **text-embedding-3-small** | Fast, cost-efficient semantic embeddings                           |
| LLM / VLM           | **gpt-4o-mini**            | Fast generation and multimodal reasoning                           |
| Application         | **Python**                 | Flexible ecosystem for document AI and data processing             |

### Why Hybrid Retrieval?

Pure vector search can struggle with identifiers where exact character matching matters.

For example:

> "What does EV-BMS-100 refer to?"

Semantic search may retrieve conceptually similar BMS content while missing the exact identifier.

Combining vector search with Tantivy enables:

**semantic understanding + deterministic lexical matching.**

---

## 4. Evaluation Results

The system was evaluated on **22 engineering question-answering cases**.

| Metric                     |              Result |
| -------------------------- | ------------------: |
| Total Evaluated            |              **22** |
| Answer Accuracy            | **21 / 22 (95.5%)** |
| Document Citation Accuracy |  **22 / 22 (100%)** |
| Page Citation Accuracy     |  **22 / 22 (100%)** |
| Not-Found Rate             |   **2 / 22 (9.1%)** |

The evaluation shows that the system was particularly strong at **grounded retrieval and provenance tracking**, achieving **100% document and page citation accuracy** across the evaluation set.

---

## 5. Failure Analysis

The primary failure cases were associated with difficult document structures rather than basic retrieval.

### 1. Complex Multi-Page Tables

Tables spanning multiple pages without clear borders can occasionally be fragmented during document parsing.

**Impact:** Information belonging to the same logical table may be indexed as separate chunks.

**Potential improvement:** Add table-aware post-processing that detects continuation tables and merges them before indexing.

### 2. Legacy Patent Layouts

Older patents containing overlapping watermarks, poor scans, or unusual layouts can reduce OCR and layout-detection quality.

**Impact:** Important text may not be extracted correctly, resulting in a retrieval miss.

**Potential improvement:** Add document-quality detection followed by adaptive OCR preprocessing, image enhancement, and alternative OCR fallback.

### 3. Not-Found Queries

Some queries were correctly identified as having insufficient evidence in the indexed corpus.

This is preferable to generating unsupported information, but the system could further improve its **retrieval confidence estimation** and distinguish between:

* Information genuinely absent from the corpus
* Information present but poorly extracted
* Information present but missed during retrieval

---

## 6. Known Limitations

* Very complex multi-page tables may require additional reconstruction logic.
* OCR quality depends on the scan resolution and document quality.
* Diagram understanding is dependent on the vision model's ability to interpret dense engineering drawings.
* Text-to-SQL generation can produce incorrect SQL for highly ambiguous natural-language questions.
* Hybrid retrieval requires careful tuning of lexical and semantic retrieval weights.
* Hosted LLM and embedding APIs are still used for generation/embedding, although document parsing and storage remain local.
* The current evaluation set contains 22 questions and should be expanded for stronger statistical confidence.

---

## 7. What I Would Do Next

With additional development time, I would focus on five areas:

### 1. Retrieval Quality

Introduce a dedicated **reranking stage** using a cross-encoder or LLM-based reranker:

```text
Hybrid Retrieval
      ↓
Top-K Candidates
      ↓
Cross-Encoder Reranking
      ↓
High-Confidence Context
```

This should improve retrieval precision for technically similar documents.

### 2. Better Multimodal Retrieval

Build a unified index containing:

* Text embeddings
* Table representations
* Figure embeddings
* OCR text
* Metadata

This would allow queries to retrieve both textual and visual evidence.

### 3. Retrieval Evaluation

Expand evaluation beyond final answer accuracy to measure:

* Recall@K
* Precision@K
* MRR / NDCG
* Context relevance
* Citation correctness
* Faithfulness / groundedness

### 4. Query & SQL Validation

Add a SQL validation layer to:

* Check generated SQL syntax
* Verify referenced columns/tables
* Automatically retry failed queries
* Prevent unsafe SQL operations

### 5. Productionization

For production deployment, I would add:

* Incremental document ingestion
* Document versioning
* Caching
* Observability/tracing
* Retrieval and generation latency metrics
* Automated evaluation pipelines
* Access control and document-level permissions

---

## 8. Summary

The Engineering RAG Engine is designed around a core principle:

> **Retrieve the right evidence first, then generate an answer that remains traceable to its source.**

By combining **layout-aware document parsing, multimodal extraction, hybrid retrieval, structured SQL querying, and page-level provenance**, the system can handle engineering knowledge that traditional text-only RAG pipelines often struggle with.

The current evaluation achieved **95.5% answer accuracy with 100% document and page citation accuracy**, providing a strong foundation for further improvements in retrieval quality, multimodal reasoning, and production scalability.
