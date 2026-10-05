import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [install|ingest|ask|eval]")
        return

    command = sys.argv[1]

    # sys.executable ensures the script always uses your active (venv) python
    python_bin = sys.executable

    if command == "install":
        print("Installing dependencies...")
        subprocess.run([python_bin, "-m", "pip", "install", "-r", "requirements.txt"])
        
    elif command == "ingest":
        print("Starting ingestion...")
        subprocess.run([python_bin, "src/ingest.py"])
        
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Error: Please provide a question.")
            print("Example: python run.py ask \"What is the voltage?\"")
            return
            
        question = sys.argv[2]
        print(f"Asking question: {question}")
        subprocess.run([python_bin, "src/query.py", question])
        
    elif command == "eval":
        print("Running evaluation...")
        subprocess.run([python_bin, "src/eval.py"])
        
    else:
        print(f"Unknown command: {command}")
        print("Available commands: install, ingest, ask, eval")

if __name__ == "__main__":
    main()