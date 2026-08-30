from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
from ocr_engine import extract_text
from statement_parser import parse_csv, parse_pdf, clean_financial_data
from nlp_model import predict_category
from db_handler import insert_transaction, initialize_db, get_connection
from dateutil import parser as dateparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
RECEIPT_DIR = BASE_DIR / "data" / "sample_receipts"
STATEMENT_DIR = BASE_DIR / "data" / "sample_statements"
MODEL_DIR = BASE_DIR / "model" / "fine_tuned_bert"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_transaction_description(row):
    parts = []
    if pd.notna(row.get('name')): parts.append(str(row['name']))
    if pd.notna(row.get('mode')): parts.append(str(row['mode']))
    if pd.notna(row.get('amount')): parts.append(f"amount {row['amount']}")
    if pd.notna(row.get('drcr')):
        drcr = str(row['drcr']).upper()
        parts.append("payment" if drcr == 'DB' else "credit received")
    return " ".join(parts) if parts else "transaction"

def normalize_date(value):
    try:
        return dateparser.parse(str(value), dayfirst=True)
    except Exception:
        return None

def categorize_transactions(df):
    try:
        df['description'] = df.apply(create_transaction_description, axis=1)
        df['predicted_category'] = df['description'].apply(predict_category)
        logger.info(f"Successfully categorized {len(df)} transactions")
        return df
    except Exception as e:
        logger.error(f"Error categorizing transactions: {str(e)}")
        df['predicted_category'] = 'Others'
        return df

def create_directories():
    for directory in (RECEIPT_DIR, STATEMENT_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


@app.before_request
def initialize_once():
    """Initialize storage once per application process (Flask 3 compatible)."""
    if not app.extensions.get("findoc_initialized"):
        create_directories()
        initialize_db()
        app.extensions["findoc_initialized"] = True
        logger.info("Application initialized")

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

MERCHANT_FILTER = "merchant = %s"
CATEGORY_FILTER = "category = %s"
SOURCE_FILTER = "source = %s"
AND = ' AND '
WHERE = ' WHERE '
# --- API Helpers ---
def parse_date_range():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    return start_date, end_date

def apply_date_filter(query, start_date, end_date):
    if start_date and end_date:
        return query + " WHERE date BETWEEN %s AND %s", [start_date, end_date]
    elif start_date:
        return query + " WHERE date >= %s", [start_date]
    elif end_date:
        return query + " WHERE date <= %s", [end_date]
    return query, []

# --- APIs ---
@app.route('/api/summary/yearly')
def yearly_summary_api():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
      SELECT EXTRACT(YEAR FROM date)::INT AS yr, category, SUM(amount)
      FROM transactions
      GROUP BY yr, category
      ORDER BY yr DESC, SUM(amount) DESC
    """)
    data = cur.fetchall()
    cur.close(); conn.close()
    years = sorted({r[0] for r in data}, reverse=True)
    result = {yr: {} for yr in years}
    for yr, cat, amt in data:
        result[yr][cat] = float(amt)
    return jsonify(result)

@app.route('/api/summary/monthly')
def monthly_summary_api():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
      SELECT EXTRACT(YEAR FROM date)::INT AS yr, EXTRACT(MONTH FROM date)::INT AS mth,
             category, SUM(amount)
      FROM transactions
      GROUP BY yr, mth, category
      ORDER BY yr DESC, mth DESC, SUM(amount) DESC
    """)
    rows = cur.fetchall()
    cur.close(); conn.close()
    result = {}
    for yr, mth, cat, amt in rows:
        result.setdefault(yr, {}).setdefault(mth, {})[cat] = float(amt)
    return jsonify(result)

@app.route('/api/summary/category')
def category_summary_api():
    conn = get_connection()
    cur = conn.cursor()
    start_date, end_date = parse_date_range()
    merchant = request.args.get('merchant')
    source = request.args.get('source')
    base_query = "SELECT category, SUM(amount) FROM transactions"
    query, params = apply_date_filter(base_query, start_date, end_date)
    filters = []
    if merchant:
        filters.append(MERCHANT_FILTER)
        params.append(merchant)
    if source:
        filters.append(SOURCE_FILTER)
        params.append(source)
    if filters:
        if 'WHERE' in query:
            query += AND + AND.join(filters)
        else:
            query += WHERE + AND.join(filters)
    query += " GROUP BY category ORDER BY SUM(amount) DESC"
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close(); conn.close()
    return jsonify({"labels": [r[0] for r in results], "values": [float(r[1]) for r in results]})

@app.route('/api/summary/daily')
def daily_summary_api():
    conn = get_connection()
    cur = conn.cursor()
    start_date, end_date = parse_date_range()
    base_query = "SELECT DATE(date), SUM(amount) FROM transactions"
    query, params = apply_date_filter(base_query, start_date, end_date)
    query += " GROUP BY DATE(date) ORDER BY DATE(date)"
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close(); conn.close()
    return jsonify({"labels": [str(r[0]) for r in results], "values": [float(r[1]) for r in results]})

@app.route('/api/summary/source')
def source_summary_api():
    conn = get_connection()
    cur = conn.cursor()
    start_date, end_date = parse_date_range()
    category = request.args.get('category')
    merchant = request.args.get('merchant')
    base_query = "SELECT source, SUM(amount) FROM transactions"
    query, params = apply_date_filter(base_query, start_date, end_date)
    filters = []
    if category:
        filters.append(CATEGORY_FILTER)
        params.append(category)
    if merchant:
        filters.append(MERCHANT_FILTER)
        params.append(merchant)
    if filters:
        if 'WHERE' in query:
            query += AND + AND.join(filters)
        else:
            query += WHERE + AND.join(filters)
    query += " GROUP BY source"
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close(); conn.close()
    return jsonify({"labels": [r[0].capitalize() for r in results], "values": [float(r[1]) for r in results]})

@app.route('/api/summary/merchant')
def merchant_summary_api():
    conn = get_connection()
    cur = conn.cursor()
    start_date, end_date = parse_date_range()
    category = request.args.get('category')
    source = request.args.get('source')
    base_query = "SELECT merchant, SUM(amount) FROM transactions"
    query, params = apply_date_filter(base_query, start_date, end_date)
    filters = []
    if category:
        filters.append(CATEGORY_FILTER)
        params.append(category)
    if source:
        filters.append(SOURCE_FILTER)
        params.append(source)
    if filters:
        if 'WHERE' in query:
            query += AND + AND.join(filters)
        else:
            query += WHERE + AND.join(filters)
    query += " GROUP BY merchant ORDER BY SUM(amount) DESC LIMIT 5"
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close(); conn.close()
    return jsonify({"labels": [r[0] if r[0] else "Unknown" for r in results], "values": [float(r[1]) for r in results]})

@app.route('/api/recent')
def recent_transactions():
    conn = get_connection()
    cur = conn.cursor()
    start_date, end_date = parse_date_range()
    category = request.args.get('category')
    merchant = request.args.get('merchant')
    source = request.args.get('source')
    search = request.args.get('search', '').strip()

    base_query = "SELECT date, name, category, amount, source, merchant FROM transactions"
    query, params = apply_date_filter(base_query, start_date, end_date)
    filters = []
    if category:
        filters.append(CATEGORY_FILTER)
        params.append(category)
    if merchant:
        filters.append(MERCHANT_FILTER)
        params.append(merchant)
    if source:
        filters.append(SOURCE_FILTER)
        params.append(source)
    if filters:
        if 'WHERE' in query:
            query += AND + AND.join(filters)
        else:
            query += WHERE + AND.join(filters)
    query += " ORDER BY date DESC LIMIT 50"
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close(); conn.close()
    # Server-side search filter (if search param provided)
    def match_search(r):
        if not search:
            return True
        s = search.lower()
        return (
            (r[1] and s in str(r[1]).lower()) or
            (r[2] and s in str(r[2]).lower()) or
            (r[4] and s in str(r[4]).lower()) or
            (r[5] and s in str(r[5]).lower()) or
            (r[3] and s in str(r[3]).lower())
        )
    filtered = [
        {"date": str(r[0]), "name": r[1], "category": r[2], "amount": float(r[3]), "source": r[4], "merchant": r[5]}
        for r in results if match_search(r)
    ]
    return jsonify(filtered)

@app.route('/upload_receipt', methods=['POST'])
def upload_receipt():
    try:
        logger.info("Receipt upload request received")
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or no file selected"}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = str(RECEIPT_DIR / filename)
        file.save(filepath)

        extracted_text = extract_text(filepath)
        if not extracted_text:
            return jsonify({"error": "No text extracted from image"}), 400

        predicted_category = predict_category(extracted_text)

        insert_transaction(
            date=datetime.now(),
            merchant="Unknown",
            amount=0,
            mode="Receipt",
            name="N/A",
            category=predicted_category,
            source="receipt",
            raw_text=extracted_text
        )

        return jsonify({
            "filename": filename,
            "text": extracted_text,
            "predicted_category": predicted_category,
            "confidence": None
        })
    except Exception as e:
        logger.error(f"Error processing receipt: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/upload_statement', methods=['POST'])
def upload_statement():
    try:
        logger.info("Statement upload request received")
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or no file selected"}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = str(STATEMENT_DIR / filename)
        file.save(filepath)

        df = parse_csv(filepath) if filename.lower().endswith(".csv") else parse_pdf(filepath)
        if df.empty:
            return jsonify({"error": "No data could be extracted from file"}), 400

        df = clean_financial_data(df)
        df = categorize_transactions(df)
        df = df.fillna('')

        inserted_count = 0
        for _, row in df.iterrows():
            parsed_date = normalize_date(row.get("date")) or datetime.now()
            try:
                insert_transaction(
                    date=parsed_date,
                    merchant=row.get("merchant") or row.get("name"),
                    amount=row.get("amount"),
                    mode=row.get("mode"),
                    name=row.get("name"),
                    category=row.get("predicted_category"),
                    source="statement",
                    raw_text=row.get("description")
                )
                inserted_count += 1
            except Exception as e:
                logger.warning(f"Skipping row due to error: {e}")

        category_summary = df['predicted_category'].value_counts().to_dict()

        return jsonify({
            "filename": filename,
            "data": df.to_dict(orient="records"),
            "row_count": inserted_count,
            "columns": list(df.columns),
            "category_summary": category_summary,
            "total_categories": len(category_summary)
        })
    except Exception as e:
        logger.error(f"Error processing statement: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e): return jsonify({"error": "File too large"}), 413

@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e): return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
