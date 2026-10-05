import os
import lancedb

def view_stored_chunks():
    # Connect to your local LanceDB database
    LANCEDB_PATH = os.path.join("db", "lancedb")
    
    try:
        db = lancedb.connect(LANCEDB_PATH)
        table = db.open_table("documents")
        
        # Load the database into a Pandas DataFrame for easy viewing
        df = table.to_pandas()
        
        # Drop the 'vector' column because it's just a massive list of numbers
        df = df.drop(columns=['vector'])
        
        print(f"\nTotal chunks stored in database: {len(df)}\n")
        print("-" * 50)
        
        # Print the first 3 chunks to the terminal
        for index, row in df.head(3).iterrows():
            print(f"📄 Document: {row['doc']} | Page: {row['page']}")
            print(f"📝 Text:\n{row['text']}")
            print("-" * 50)
            
        # Export everything to a CSV so you can read all chunks easily
        export_path = "db/all_extracted_chunks.csv"
        df.to_csv(export_path, index=False)
        print(f"\n✅ All {len(df)} chunks have been saved to '{export_path}'")
        print("You can open this CSV file in VS Code or Excel to inspect how your text, tables, and figures were chunked!")
        
    except Exception as e:
        print(f"Error reading database: {e}")
        print("Make sure you have successfully run 'python run.py ingest' first.")

if __name__ == "__main__":
    view_stored_chunks()