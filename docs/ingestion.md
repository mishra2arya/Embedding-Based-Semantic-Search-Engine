# Document Ingestion & Chunking System

## Ingestion Pipeline Flow
```text
Raw File / API Payload
         ↓
Input Validation (Size limits, path sanitization)
         ↓
Parser Factory (TXT, MD, HTML, JSON, CSV, PDF, DOCX)
         ↓
Unicode NFKC & Whitespace Normalization
         ↓
Content Hash (SHA-256) & Deduplication Filter
         ↓
Deterministic Document ID (`doc_<checksum[:16]>`)
         ↓
Dynamic Chunking Strategy
         ↓
Embedding Generation & FAISS / SQLite Insertion
```

## Supported Parsers
| Format | Parser | Features |
| :--- | :--- | :--- |
| `.txt`, `.md` | `TextParser` | Header extraction, frontmatter stripping |
| `.html`, `.htm` | `HTMLParser` | Title extraction, script/style tag decomposition |
| `.json`, `.jsonl`| `JSONParser` | Structured key mapping, batch JSONL streaming |
| `.csv` | `CSVParser` | Column-header aligned row serialization |
| `.pdf` | `PDFParser` | Multi-page text extraction, author & title metadata |
| `.docx` | `DOCXParser` | Paragraph and section extraction |

## Dynamic Chunking Strategies
1. **Token Chunking (`token`)**: Fixed sliding window of words/tokens with configurable overlap.
2. **Sentence Chunking (`sentence`)**: Splits along sentence boundary punctuation (`.`, `!`, `?`), grouping sentences until target token budget is reached.
3. **Semantic Chunking (`semantic`)**: Evaluates cosine distance between consecutive sentence embeddings; splits when topic shift exceeds similarity threshold.
4. **Structure-Aware Chunking (`structure`)**: Respects markdown headers (`#`), code blocks (```` ``` ````), tables, and paragraph breaks.
