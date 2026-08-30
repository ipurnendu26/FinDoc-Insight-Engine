# Financial Document Intelligence

A portfolio application for extracting and organizing transaction data from receipt images and CSV/PDF statements. It combines OCR, document parsing, NLP-based category prediction, PostgreSQL persistence and a browser dashboard.

## Implemented workflow

1. Upload a receipt image or CSV/PDF statement.
2. Validate the file type and size.
3. Extract text with Tesseract or parse tabular statement data.
4. Normalize dates and numeric transaction fields.
5. Categorize transaction descriptions with a BERT classifier.
6. Store normalized records in PostgreSQL.
7. Query yearly, monthly, daily, merchant, source and category summaries through Flask endpoints.

## Scope and evidence

The repository contains the application source, HTML templates and model-training code. Trained BERT weights and private financial documents are intentionally not committed. When no saved model is present, the current implementation fine-tunes from built-in illustrative category examples on first use. That behavior demonstrates integration, but it is not a production-quality or independently validated financial classifier.

Receipt uploads currently store an amount of zero and an unknown merchant unless those fields are parsed from a statement. The API therefore does not report a fabricated confidence value.

## Technology

Python, Flask, TensorFlow, Hugging Face Transformers, Tesseract, OpenCV, pandas, PostgreSQL, Chart.js, Docker and GitHub Actions.

## Run with Docker

    git clone https://github.com/ipurnendu26/FinDoc-Insight-Engine.git
    cd FinDoc-Insight-Engine
    docker compose up --build

Open http://localhost:5000. The compose file starts PostgreSQL and the Flask application with local-development credentials.

## Run locally

1. Install Tesseract and Poppler for your operating system.
2. Create and activate a Python 3.10 virtual environment.
3. Install dependencies with pip install -r requirements.txt.
4. Copy .env.example to .env and set PostgreSQL values.
5. Start PostgreSQL.
6. Run python app/main.py.

Database settings come from environment variables; credentials are not hardcoded in source.

## Quality checks

    pip install -r requirements-dev.txt
    pytest -q
    python -m compileall app

CI runs parsing tests and Python syntax checks. Additional integration tests should use disposable PostgreSQL and model fixtures.

## Repository structure

- app/main.py — routes, upload orchestration and analytics APIs
- app/ocr_engine.py — image preprocessing and Tesseract extraction
- app/statement_parser.py — CSV/PDF parsing and normalization
- app/nlp_model.py — lazy-loaded BERT classification workflow
- app/db_handler.py — environment-configured PostgreSQL access
- templates/ — upload and analytics dashboard views
- tests/ — deterministic data-cleaning tests

## Security and limitations

Do not upload real financial records to an untrusted deployment. The project does not implement authentication, authorization, encryption-at-rest policy, malware scanning or a production secrets manager. Model performance must be evaluated on a representative labeled dataset before any operational use.

## License

MIT
