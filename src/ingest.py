import os
import glob
import base64
from io import BytesIO
import pandas as pd
import duckdb
import lancedb
from dotenv import load_dotenv
from openai import OpenAI
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.chunking import HierarchicalChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption

# Load OpenAI Client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "data"
DB_DIR = "db"
DUCKDB_PATH = os.path.join(DB_DIR, "structured_data.duckdb")
LANCEDB_PATH = os.path.join(DB_DIR, "lancedb")

os.makedirs(DB_DIR, exist_ok=True)

def get_embedding(text):
    """Generates a fast vector embedding using OpenAI."""
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def analyze_image_with_vlm(pil_image):
    """Passes an image to OpenAI's GPT-4o-mini vision model."""
    try:
        # Convert the PIL image to a base64 string in memory (super fast)
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        prompt = """
        Analyze this engineering block diagram or figure. 
        Extract a structured list mapping every reference numeral to its corresponding component name.
        Note any directional flow or arrows. Keep it concise.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"VLM Analysis failed: {e}")
        return "Figure content could not be extracted."

def ingest_structured_data():
    """Ingests CSV and XLSX files into DuckDB."""
    print("\n--- Ingesting Structured Data ---")
    conn = duckdb.connect(DUCKDB_PATH)
    
    for csv_file in glob.glob(f"{DATA_DIR}/*.csv"):
        table_name = os.path.basename(csv_file).replace('.csv', '').replace('-', '_')
        print(f"Loading {csv_file} into DuckDB...")
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM read_csv_auto('{csv_file}')")
        
    for xlsx_file in glob.glob(f"{DATA_DIR}/*.xlsx"):
        table_name = os.path.basename(xlsx_file).replace('.xlsx', '').replace('-', '_')
        print(f"Loading {xlsx_file} into DuckDB...")
        df = pd.read_excel(xlsx_file, engine='openpyxl')
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        
    conn.close()
    print("Structured data ingestion complete.")

def ingest_unstructured_data():
    """Extracts text and tables using semantic chunking, and analyzes figures via VLM."""
    print("\n--- Ingesting Unstructured PDFs ---")
    db = lancedb.connect(LANCEDB_PATH)
    data_to_insert = []
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True 
    pipeline_options.generate_picture_images = True 
    
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.DOCX: WordFormatOption() 
        }
    )
    
    from docling.chunking import HierarchicalChunker
    chunker = HierarchicalChunker()
    
    # 2. Glob both PDFs and DOCX files
    target_files = glob.glob(f"{DATA_DIR}/*.pdf") + glob.glob(f"{DATA_DIR}/*.docx")
    
    if not target_files:
        print("No documents found.")
        return

    # 3. Loop through the combined list of files
    for file_path in target_files:
        doc_name = os.path.basename(file_path)
        print(f"Processing (Semantic Chunking & VLM): {doc_name}...")
        
        result = converter.convert(file_path)
        
        # PASS 1: Intelligently chunk Text and Tables with OVERLAP
        chunks = chunker.chunk(result.document)
        
        previous_chunk_text = "" # Variable to hold our overlap memory
        
        for chunk in chunks:
            raw_text = chunk.text
            
            if not raw_text or len(raw_text.strip()) < 10:
                continue
                
            # --- OVERLAP LOGIC ---
            # Take the last 300 characters of the previous chunk and prepend it
            overlap_text = previous_chunk_text[-300:] if previous_chunk_text else ""
            
            if overlap_text:
                text_content = f"...{overlap_text}\n{raw_text}"
            else:
                text_content = raw_text
                
            # Store the current raw text to use as overlap for the next iteration
            previous_chunk_text = raw_text
            # ---------------------
                
            # FIX: Safely extract page number from the chunk's underlying document items
            page_num = 1
            if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'prov') and item.prov:
                        page_num = item.prov[0].page_no
                        break
                
            vector = get_embedding(text_content)
            
            data_to_insert.append({
                "vector": vector,
                "text": text_content,
                "doc": doc_name,
                "page": page_num
            })
            
        # PASS 2: Handle Figures with OpenAI Vision
        for pic in result.document.pictures:
            captions = [cap.text for cap in getattr(pic, "captions", []) if hasattr(cap, "text")]
            caption_str = " ".join(captions)
            
            print(f"Found figure, analyzing with GPT-4o-mini...")
            try:
                pil_image = pic.get_image(result.document)
                vlm_description = analyze_image_with_vlm(pil_image)
                text_content = f"[FIGURE] Caption: {caption_str}\n[VISUAL CONTENT]:\n{vlm_description}"
            except Exception:
                text_content = f"[FIGURE] Caption: {caption_str}"
                
            page_num = 1
            if hasattr(pic, 'prov') and pic.prov:
                page_num = pic.prov[0].page_no
                
            vector = get_embedding(text_content)
            
            data_to_insert.append({
                "vector": vector,
                "text": text_content,
                "doc": doc_name,
                "page": page_num
            })
            
    if data_to_insert:
        print(f"Saving {len(data_to_insert)} chunks to LanceDB...")
        db.create_table("documents", data=data_to_insert, mode="overwrite")
        
        # --- NEW CODE: Build the Hybrid Search Index ---
        print("Building Full-Text Search (FTS) index for Hybrid Search...")
        table = db.open_table("documents")
        table.create_fts_index("text")
        # -----------------------------------------------
        
        print("Unstructured data ingestion complete.")

if __name__ == "__main__":
    ingest_structured_data()
    ingest_unstructured_data()