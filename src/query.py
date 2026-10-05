import sys
import os
import json
import duckdb
import lancedb
from dotenv import load_dotenv
from openai import OpenAI

# Load OpenAI Client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_DIR = "db"
LANCEDB_PATH = os.path.join(DB_DIR, "lancedb")
DUCKDB_PATH = os.path.join(DB_DIR, "structured_data.duckdb")
MODEL_NAME = "gpt-4o-mini"


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def classify_query(query):
    """Routes the question to the SQL engine or the Vector engine."""
    prompt = f"""
    You are a classification router for an engineering database.
    Determine if the following question requires querying a structured spreadsheet database (SQL) or an unstructured document database (Vector Search).

    Route to "STRUCTURED" ONLY if the question explicitly asks about:
    - The Bill of Materials (BOM)
    - Test logs, test runs, or pass/fail metrics
    - Calculating costs, tabular quantities, or specific spreadsheet part numbers.

    Route to "UNSTRUCTURED" if the question asks about:
    - Patents, claims, or specifications (e.g., US-XXXX patents, UAV VTOL)
    - Figures, diagrams, reference numerals, or images
    - Engineering design documents, materials, dimensions, or general text.

    Question: "{query}"
    Reply with ONLY the word "STRUCTURED" or "UNSTRUCTURED". Do not add any punctuation.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip().upper()


import glob


def handle_structured_query(query):
    """Executes SQL and correctly maps the result to the original .csv or .xlsx filename."""
    conn = duckdb.connect(DUCKDB_PATH)

    # Scan the data folder to map table names back to exact filenames
    file_map = {}

    for filepath in glob.glob("data/*.*"):
        filename = os.path.basename(filepath)
        table = os.path.splitext(filename)[0]
        file_map[table] = filename

    tables = conn.execute("SHOW TABLES").fetchall()
    schema_info = ""

    for table in tables:
        table_name = table[0]
        exact_source_file = file_map.get(
            table_name,
            f"{table_name}.csv"
        )

        columns = conn.execute(
            f"DESCRIBE {table_name}"
        ).fetchall()

        schema_info += (
            f"Table: {table_name} "
            f"(Source File: {exact_source_file})\n"
            f"Columns: {[col[0] for col in columns]}\n\n"
        )

    sql_prompt = f"""
    You are an expert DuckDB SQL developer.
    Database schemas:
    {schema_info}

    Write a SQL query to answer: "{query}"

    CRITICAL SQL RULES:
    1. USE EXACT COLUMN NAMES from the schema.
    2. STRING MATCHING: NEVER use '=' for strings. ALWAYS use ILIKE with wildcards (e.g., WHERE result ILIKE '%fail%') to prevent whitespace/case errors.
    3. AGGREGATIONS: Use SELECT COUNT(...) and alias it if asked for counts.
    4. DISTINCT: Use COUNT(DISTINCT col) if asked for distinct units.
    5. CITATION: You MUST use the exact "Source File" provided in the schema block for the "source_file" JSON field.

    Respond ONLY in valid JSON matching this exact schema:
    {{
        "sql": "SELECT * FROM ...",
        "source_file": "exact_file_name_from_schema"
    }}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': sql_prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    try:
        response_data = json.loads(
            response.choices[0].message.content.strip()
        )

        sql_query = (
            response_data.get("sql", "")
            .replace("```sql", "")
            .replace("```", "")
            .strip()
        )

        source_file = response_data.get(
            "source_file",
            "structured_data_file"
        )

        cursor = conn.execute(sql_query)

        columns = [desc[0] for desc in cursor.description]
        records = cursor.fetchall()

        formatted_result = [
            dict(zip(columns, row))
            for row in records
        ]

        # FIX:
        # Add an explicit citation tag so the final LLM
        # knows the exact source filename and page.
        context = (
            f"[SOURCE: {source_file}, PAGE: 1]\n"
            f"SQL Query Result: {formatted_result}\n"
            f"---\n"
        )

    except Exception as e:
        source_file = (
            response_data.get("source_file", "unknown")
            if 'response_data' in locals()
            else "unknown"
        )

        context = (
            f"\nSQL Execution Failed: {e}\n"
            f"---\n"
        )

    conn.close()
    return context


def handle_unstructured_query(query):
    """Executes Hybrid Search and retrieves the top 15 chunks to ensure maximum context."""
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table("documents")

    query_vector = get_embedding(query)

    try:
        # Retrieve 15 chunks to capture wide tables and separated context
        results = (
            table
            .search(query, query_type="hybrid")
            .vector(query_vector)
            .limit(15)
            .to_list()
        )
    except Exception:
        results = (
            table
            .search(query_vector)
            .limit(15)
            .to_list()
        )

    if not results:
        return "none"

    context = ""

    for r in results:
        context += (
            f"[SOURCE: {r['doc']}, PAGE: {r['page']}]\n"
            f"{r['text']}\n"
            f"---\n"
        )

    return context


def ask_eval_question(query):
    route = classify_query(query)

    if route == "STRUCTURED":
        context = handle_structured_query(query)
    else:
        context = handle_unstructured_query(query)

    final_prompt = f"""
    You are an expert engineering answering assistant.

    RULES:
    1. Read the Context thoroughly. Look for synonyms, related terms, or overlapping concepts.
    2. NEVER TRUNCATE SPECIFICATIONS. If a component rating has multiple conditions, times, or peak limits (e.g., "320 A continuous, 450 A for 10 seconds"), you MUST include the entire phrase. Extract all associated conditions and comma-separated values.
    3. Before giving the final answer, write down your reasoning step-by-step in the "reasoning" field.

    CITATION RULES:
    4. Every citation MUST come directly from a [SOURCE: ..., PAGE: ...] tag in the Context.
    5. NEVER use "SQL Query Result", "source_doc_name", "structured_data_file", "unknown", or any other placeholder as the document name.
    6. For STRUCTURED queries, the document name MUST be the exact filename present in the [SOURCE: ...] tag.
    7. For STRUCTURED queries, the page number MUST be 1.
    8. Do not infer, invent, rename, or modify the source filename.
    9. If the answer comes from the SQL result, cite the exact source file shown in the citation tag.
    10. If multiple source tags are present, cite only the sources actually used to answer the question.

    Format your response in STRICT JSON matching this schema exactly:
    {{
        "reasoning": "...",
        "answer": "your comprehensive final answer",
        "citations": [
            {{
                "doc": "EXACT filename from [SOURCE: ..., PAGE: ...]",
                "page": 1
            }}
        ],
        "not found": false
    }}

    Context:
    {context}

    Question: {query}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{'role': 'user', 'content': final_prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    try:
        result_dict = json.loads(
            response.choices[0].message.content.strip()
        )

        return route, result_dict

    except Exception as e:
        return route, {
            "error": str(e),
            "answer": "format error",
            "citations": []
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a question.")
    else:
        ask_question(sys.argv[1])