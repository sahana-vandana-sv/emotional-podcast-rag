# Test local:    python scripts/test_api.py
# Test deployed: python scripts/test_api.py --url https://your-app.onrender.com

import sys, argparse
import requests

def test_api(base_url: str):
    base_url = base_url.rstrip("/")
    print(f"\nTESTING: {base_url}\n")
    results = []

    # Test 1: Root
    r = requests.get(f"{base_url}/", timeout=10)
    passed = r.status_code == 200
    print(f"{'✅' if passed else '❌'}  GET /  ({r.status_code})")
    results.append(passed)

    # Test 2: Health
    r = requests.get(f"{base_url}/health", timeout=10)
    data = r.json()
    passed = r.status_code == 200 and data["episodes_loaded"] > 0
    print(f"{'✅' if passed else '❌'}  GET /health — episodes: {data.get('episodes_loaded', 0)}")
    results.append(passed)

    # Test 3: Valid search
    r = requests.post(f"{base_url}/api/search",
        json={"query": "I feel stressed about my job", "num_recommendations": 2},
        timeout=30)
    data = r.json()
    passed = r.status_code == 200 and len(data.get("recommendations", [])) > 0
    print(f"{'✅' if passed else '❌'}  POST /api/search — emotion: {data.get('primary_emotion', 'N/A')}")
    results.append(passed)

    # Test 4: Empty query
    r = requests.post(f"{base_url}/api/search", json={"query": ""}, timeout=10)
    passed = r.status_code in [422, 400]
    print(f"{'✅' if passed else '❌'}  POST /api/search (empty) — correctly rejected ({r.status_code})")
    results.append(passed)

    # Test 5: Clear memory
    r = requests.delete(f"{base_url}/api/memory/clear", timeout=10)
    passed = r.status_code == 200
    print(f"{'✅' if passed else '❌'}  DELETE /api/memory/clear")
    results.append(passed)

    print(f"\n{sum(results)}/{len(results)} tests passed\n")
    return all(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    sys.exit(0 if test_api(args.url) else 1)


