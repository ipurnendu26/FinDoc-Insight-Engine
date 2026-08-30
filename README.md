# Financial Document Intelligence Engine

An end-to-end application for extracting, normalizing, categorizing, storing, and analyzing information from receipts and financial statements.

The project demonstrates how OCR, document parsing, NLP, relational storage, and interactive analytics can be combined into one auditable workflow.

## Problem

Financial records arrive in inconsistent formats: receipt images, PDFs, and CSV statements. Manual entry is slow and error-prone. This system converts those documents into normalized transaction records that can be searched, categorized, visualized, and exported.

## Processing workflow

```text
Receipt image / PDF / CSV
    -> secure upload validation
    -> OCR or statement parsing
    -> field normalization
    -> expense-category prediction
    -> PostgreSQL persistence
    -> filtering, analytics, and export
```

## Capabilities

- Receipt image, PDF statement, and CSV statement ingestion
- OCR-based text extraction
- Transaction date, merchant, amount, payment mode, and source normalization
- BERT-based expense-category classification
- PostgreSQL transaction persistence
- Search and filters across dates, merchants, categories, and sources
- Spending summaries and downloadable CSV/Excel reports
- Flask-based application interface

## Technology stack

| Component | Technology |
|---|---|
| Web application | Flask |
| OCR | EasyOCR, OpenCV |
| NLP | BERT, Hugging Face Transformers, TensorFlow |
| Data processing | Pandas |
| PDF and statement parsing | pdfminer, PyPDF2, Camelot, Tabula |
| Database | PostgreSQL, psycopg2 |
| Visualization | Matplotlib and application templates |

## Repository structure

```text
.
├── app/
│   ├── main.py
│   ├── ocr_engine.py
│   ├── statement_parser.py
│   ├── nlp_model.py
│   └── db_handler.py
├── templates/
├── requirements.txt
└── README.md
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env
```

Create the PostgreSQL database, update the environment variables in `.env`, and run:

```bash
python app/main.py
```

Database credentials are read from environment variables and should never be committed.

## Model transparency

The current repository creates a balanced category-training corpus from curated example transaction descriptions. This is useful for demonstrating the fine-tuning workflow, but it is not a substitute for evaluation on independently collected, human-labeled financial transactions.

A production-quality evaluation should report:

- Macro and weighted F1
- Per-category precision and recall
- Confusion matrix
- Performance on unseen merchants and wording
- Confidence calibration
- OCR-to-category error propagation

The README intentionally does not claim production accuracy until reproducible external evaluation artifacts are included.

## Data and security considerations

- Use synthetic or appropriately sanitized financial documents for demonstrations.
- Do not commit uploaded statements, database dumps, credentials, or trained artifacts containing sensitive data.
- Apply file-size, extension, and content validation before deployment.
- Protect production endpoints with authentication, authorization, CSRF controls, encryption, and audit logging.
- Review OCR and classification outputs before relying on extracted financial values.

## Current limitations

- Training examples are curated within the codebase.
- Formal external classification and OCR benchmarks are not yet included.
- Automated tests and containerized deployment are planned.
- The application is a portfolio engineering project, not financial advice or production accounting software.

## Roadmap

- Add labeled evaluation fixtures and reproducible metric reports
- Add unit tests for parsing, normalization, and database operations
- Add Docker Compose for Flask and PostgreSQL
- Add confidence-aware human review
- Add schema migrations and role-based access controls
- Add document-level provenance and extraction confidence

## Author

**Purnendu Kale**  
[LinkedIn](https://www.linkedin.com/in/purnendukale/) · [GitHub](https://github.com/ipurnendu26)
