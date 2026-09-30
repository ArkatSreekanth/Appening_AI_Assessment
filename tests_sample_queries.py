import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import PINECONE_INDEX_NAME
from src.graph import build_rag_graph

API_URL = "http://127.0.0.1:8000/chat"

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "expected_refusal": False,
        "description": "Definition and concept test"
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "expected_refusal": False,
        "description": "Comparative understanding test"
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "expected_refusal": False,
        "description": "Architectural breakdown test"
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "expected_refusal": False,
        "description": "Deep component workflow test"
    },
    {
        "id": 5,
        "query": "Who won the 2022 FIFA World Cup?",
        "expected_refusal": True,
        "description": "Strict grounding validation test (Out-of-domain refusal)"
    }
]


def run_via_api(query: str):
    """Execute query via FastAPI endpoint."""
    response = requests.post(API_URL, json={"query": query}, timeout=60)
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data["final_answer"],
        "context": data["retrieved_context"],
        "score": data["confidence_score"]
    }


def run_via_direct_graph(query: str, graph):
    """Execute query directly via compiled LangGraph."""
    initial_state = {
        "question": query,
        "context": [],
        "answer": "",
        "score": 0.0
    }
    result = graph.invoke(initial_state)
    return {
        "answer": result["answer"],
        "context": result["context"],
        "score": result["score"]
    }


def main():
    print("=" * 80)
    print("🤖 Agentic AI Chatbot — Benchmark Evaluation & Verification Suite")
    print("=" * 80)

    # Determine execution mode: API if running, else direct LangGraph
    use_api = False
    try:
        health_resp = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if health_resp.status_code == 200:
            use_api = True
            print("[Mode] Running tests against live FastAPI server at http://127.0.0.1:8000")
    except Exception:
        pass

    graph = None
    if not use_api:
        print("[Mode] FastAPI server not detected. Initializing compiled LangGraph directly...")
        graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)

    results_summary = []

    for test in BENCHMARK_QUERIES:
        q_id = test["id"]
        query = test["query"]
        expected_refusal = test["expected_refusal"]
        desc = test["description"]

        print("\n" + "-" * 80)
        print(f"Test #{q_id}: [{desc}]")
        print(f"Question: \"{query}\"")
        print("-" * 80)

        start_time = time.time()
        try:
            if use_api:
                res = run_via_api(query)
            else:
                res = run_via_direct_graph(query, graph)

            elapsed = time.time() - start_time
            answer = res["answer"]
            chunks = res["context"]
            score = res["score"]

            print(f"\n[Generated Answer]:\n{answer}\n")
            print(f"[Grounding Score]: {score:.2f}")
            print(f"[Retrieved Chunks]: {len(chunks)} chunks retrieved (Latency: {elapsed:.2f}s)")
            if chunks:
                print(f"[First Chunk Sample]: {chunks[0][:150]}...")

            # Verification Logic
            refusal_markers = [
                "cannot answer based on the provided document",
                "not mentioned",
                "does not contain",
                "not provided"
            ]
            is_refused = any(m in answer.lower() for m in refusal_markers)

            passed = False
            if expected_refusal:
                if is_refused and score == 0.0:
                    passed = True
                    status_note = "PASSED (Correctly Refused Out-of-Domain Query)"
                else:
                    status_note = "FAILED (Should have refused out-of-domain query)"
            else:
                if not is_refused and score >= 0.8:
                    passed = True
                    status_note = "PASSED (Grounded Answer with High Confidence)"
                else:
                    status_note = "FAILED (Incomplete grounding or low confidence)"

            print(f"[Status]: {status_note}")
            results_summary.append({
                "id": q_id,
                "passed": passed,
                "note": status_note,
                "latency": f"{elapsed:.2f}s"
            })

        except Exception as e:
            print(f"[ERROR]: Failed to execute query: {str(e)}")
            results_summary.append({
                "id": q_id,
                "passed": False,
                "note": f"ERROR: {str(e)}",
                "latency": "N/A"
            })

    # Final Summary Report
    print("\n" + "=" * 80)
    print("📊 BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    all_passed = True
    for r in results_summary:
        icon = "✅" if r["passed"] else "❌"
        if not r["passed"]:
            all_passed = False
        print(f"Test #{r['id']}: {icon} {r['note']} (Latency: {r['latency']})")

    print("-" * 80)
    if all_passed:
        print("🎉 ALL BENCHMARK VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("⚠️ Some benchmark tests had warnings or failures. Review above output.")
    print("=" * 80)


if __name__ == "__main__":
    main()
