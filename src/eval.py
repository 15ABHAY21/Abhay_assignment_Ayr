import os
import sys
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

# Import retrieval & answering functions from query.py
from query import classify_query, handle_structured_query, handle_unstructured_query, MODEL_NAME

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GOLD_SET_PATH = os.path.join("data", "gold_set.json")

# Fallback gold set if data/gold_set.json does not exist
DEFAULT_GOLD_SET = [
    {
    "id": "q15",
    "type": "figure",
    "question": "In FIG. 3 of US2017/0320570A1, what is component 18?",
    "reference_answer": "The left thrust-generating device.",
    "evidence": [{"doc": "US20170320570A1_uav_vtol.pdf", "page": 1}]
  }

]

def load_gold_set():
    """Loads gold set from JSON file if available, otherwise uses DEFAULT_GOLD_SET."""
    if os.path.exists(GOLD_SET_PATH):
        with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists("gold_set.json"):
        with open("gold_set.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_GOLD_SET

def execute_engine(query):
    """Executes the pipeline and returns the structured dictionary response."""
    route = classify_query(query)
    if route == "STRUCTURED":
        context = handle_structured_query(query)
    else:
        context = handle_unstructured_query(query)

    prompt = f"""
    You are an expert engineering answering assistant. Answer using ONLY the Context below.
    If the context does not contain the answer, set "not found" to true.

    RULES:
    1. Look for synonyms, component identifiers, or related claims.
    2. Document step-by-step logic in the "reasoning" key.
    3. Ensure citations match [SOURCE: ..., PAGE: ...] tags in the context.

    Format your response in STRICT JSON:
    {{
      "reasoning": "step-by-step extraction logic",
      "answer": "concise factual answer",
      "citations": [{{"doc": "source_doc_name", "page": 1}}],
      "not found": false
    }}

    Context:
    {context}

    Question: {query}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        return {"reasoning": "", "answer": f"Parse error: {e}", "citations": [], "not found": True}

def evaluate_answer_with_llm(question, reference_answer, generated_answer):
    """Uses LLM-as-a-judge to evaluate semantic equivalence between reference and generated answer."""
    eval_prompt = f"""
    You are an impartial evaluator for technical question answering.
    
    Question: {question}
    Reference Answer: {reference_answer}
    Generated Answer: {generated_answer}
    
    Task: Determine if the Generated Answer captures the core factual truth stated in the Reference Answer.
    - Minor phrasing, article differences, or extra supporting context are acceptable.
    - Contradictions, missing key identifiers/numbers, or "not found" statements fail.
    
    Respond in STRICT JSON:
    {{
      "is_correct": true or false,
      "reason": "one sentence explanation"
    }}
    """
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": eval_prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        data = json.loads(res.choices[0].message.content.strip())
        return bool(data.get("is_correct", False)), data.get("reason", "")
    except Exception as e:
        # Fallback to normalized substring inclusion
        passed = any(tok in generated_answer.lower() for tok in reference_answer.lower().split() if len(tok) > 4)
        return passed, f"Fallback check due to error: {e}"

def evaluate_citations(evidence_list, actual_citations):
    """
    Compares generated citations against ground-truth evidence objects.
    Checks document name alignment and page accuracy.
    """
    # 1. NEW LOGIC: If both are empty (e.g., a "not found" query), it is a perfect match!
    if not evidence_list and not actual_citations:
        return True, True
        
    # 2. If one is empty but the other is not, it is a mismatch failure.
    if not evidence_list or not actual_citations:
        return False, False

    doc_matched = False
    page_matched = False

    for ev in evidence_list:
        target_doc = os.path.basename(ev.get("doc", "")).lower()
        target_page = ev.get("page")

        for cit in actual_citations:
            cited_doc = os.path.basename(cit.get("doc", "")).lower()
            cited_page = cit.get("page")

            # Match filename by equality or clean substring
            if target_doc in cited_doc or cited_doc in target_doc:
                doc_matched = True
                if cited_page is not None and target_page is not None:
                    # Allow exact or +/-1 tolerance for OCR/header pagination shifts
                    if abs(int(cited_page) - int(target_page)) <= 1:
                        page_matched = True
                        break

    return doc_matched, page_matched

def run_evaluation():
    gold_set = load_gold_set()
    total = len(gold_set)
    print(f"\n================ STARTING EVALUATION ({total} Items) ================\n")

    correct_answers = 0
    correct_docs = 0
    correct_pages = 0
    not_found_count = 0

    for idx, item in enumerate(gold_set, 1):
        q_id = item.get("id", f"q{idx}")
        q_type = item.get("type", "factual")
        question = item["question"]
        ref_answer = item["reference_answer"]
        evidence = item.get("evidence", [])

        print(f"[{q_id}] ({q_type}) {question}")

        # Run system pipeline
        result = execute_engine(question)
        ans = result.get("answer", "")
        cits = result.get("citations", [])
        is_not_found = result.get("not found", False)

        if is_not_found:
            not_found_count += 1

        # Evaluate Answer
        is_correct, reason = evaluate_answer_with_llm(question, ref_answer, ans)
        if is_correct:
            correct_answers += 1

        # Evaluate Evidence Citations
        doc_ok, page_ok = evaluate_citations(evidence, cits)
        if doc_ok:
            correct_docs += 1
        if page_ok:
            correct_pages += 1

        # Console logging
        print(f"  • Reference: {ref_answer}")
        print(f"  • Generated: {ans}")
        print(f"  • Answer Judge:   {'✅ PASS' if is_correct else '❌ FAIL'} ({reason})")
        print(f"  • Doc Citation:   {'✅ PASS' if doc_ok else '❌ FAIL'} (Target: {[e['doc'] for e in evidence]}, Got: {[c.get('doc') for c in cits]})")
        print(f"  • Page Citation:  {'✅ PASS' if page_ok else '❌ FAIL'} (Target: {[e['page'] for e in evidence]}, Got: {[c.get('page') for c in cits]})\n")

        time.sleep(0.5)

    print("================== SUMMARY RESULTS ==================")
    print(f"Total Evaluated:        {total}")
    print(f"Answer Accuracy:        {correct_answers}/{total} ({correct_answers/total * 100:.1f}%)")
    print(f"Document Citation Acc:  {correct_docs}/{total} ({correct_docs/total * 100:.1f}%)")
    print(f"Page Citation Acc:      {correct_pages}/{total} ({correct_pages/total * 100:.1f}%)")
    print(f"Not-Found Rate:         {not_found_count}/{total} ({not_found_count/total * 100:.1f}%)")
    print("=====================================================\n")

if __name__ == "__main__":
    run_evaluation()