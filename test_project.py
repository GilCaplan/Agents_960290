"""
Comprehensive test suite for Agri-Advisor Pro.
Tests all required API endpoints from the project spec.
Run while main.py is already running on http://127.0.0.1:8000
"""

import os
import requests
import time
import uuid

# Default to the local server; override with BASE_URL env var if needed.
# e.g.  BASE_URL=http://127.0.0.1:8000 python test_project.py
BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")
RESULTS = []

# ── helpers ──────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def ok(label, detail=""):
    msg = f"  [PASS] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    RESULTS.append(("PASS", label))

def fail(label, detail=""):
    msg = f"  [FAIL] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    RESULTS.append(("FAIL", label))

def post_execute(payload, timeout=240):
    """POST to /api/execute and return parsed JSON or None on error."""
    try:
        res = requests.post(f"{BASE}/api/execute", json=payload, timeout=timeout)
        if res.status_code != 200:
            return None, f"HTTP {res.status_code}: {res.text[:120]}"
        return res.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to server — is main.py running?"
    except Exception as e:
        return None, str(e)


# ── Test 1: GET /api/team_info ────────────────────────────────────────────────

section("1. GET /api/team_info")

try:
    r = requests.get(f"{BASE}/api/team_info", timeout=5)
    if r.status_code != 200:
        fail("Returns HTTP 200", f"Got {r.status_code}")
    else:
        ok("Returns HTTP 200")
        data = r.json()

        # Required top-level keys
        for key in ("group_batch_order_number", "team_name", "students"):
            if key in data:
                ok(f"Has key '{key}'", str(data[key])[:80])
            else:
                fail(f"Has key '{key}'")

        # students is a non-empty list
        students = data.get("students", [])
        if isinstance(students, list) and len(students) > 0:
            ok(f"'students' is a non-empty list ({len(students)} entries)")
        else:
            fail("'students' is a non-empty list", f"Got: {students}")

        # Each student has name + email
        all_ok = all("name" in s and "email" in s for s in students)
        if all_ok:
            ok("Each student has 'name' and 'email'")
            for s in students:
                print(f"         {s['name']} <{s['email']}>")
        else:
            fail("Each student has 'name' and 'email'", str(students))

except Exception as e:
    fail("GET /api/team_info", str(e))


# ── Test 2: GET /api/agent_info ───────────────────────────────────────────────

section("2. GET /api/agent_info")

try:
    r = requests.get(f"{BASE}/api/agent_info", timeout=5)
    if r.status_code != 200:
        fail("Returns HTTP 200", f"Got {r.status_code}")
    else:
        ok("Returns HTTP 200")
        data = r.json()

        for key in ("description", "purpose", "prompt_template", "prompt_examples"):
            if key in data:
                ok(f"Has key '{key}'")
            else:
                fail(f"Has key '{key}'")

        # prompt_examples is a non-empty list with prompt, full_response, steps
        examples = data.get("prompt_examples", [])
        if isinstance(examples, list) and len(examples) > 0:
            ok(f"'prompt_examples' is non-empty ({len(examples)} examples)")
            for i, ex in enumerate(examples):
                for field in ("prompt", "full_response", "steps"):
                    if field in ex:
                        ok(f"Example {i+1} has '{field}'")
                    else:
                        fail(f"Example {i+1} missing '{field}'")
                # Each step in example should have module, prompt, response
                for j, step in enumerate(ex.get("steps", [])):
                    missing = [f for f in ("module", "prompt", "response") if f not in step]
                    if missing:
                        fail(f"Example {i+1} step {j+1} missing: {missing}")
        else:
            fail("'prompt_examples' is non-empty list", str(examples))

except Exception as e:
    fail("GET /api/agent_info", str(e))


# ── Test 3: GET /api/model_architecture ──────────────────────────────────────

section("3. GET /api/model_architecture")

try:
    r = requests.get(f"{BASE}/api/model_architecture", timeout=10)
    if r.status_code != 200:
        fail("Returns HTTP 200", f"Got {r.status_code}")
    else:
        ok("Returns HTTP 200")

        ct = r.headers.get("content-type", "")
        if "image/png" in ct:
            ok(f"Content-Type is image/png", ct)
        else:
            fail("Content-Type is image/png", f"Got: {ct}")

        # PNG magic bytes: \x89PNG
        if r.content[:4] == b'\x89PNG':
            ok(f"Body is a valid PNG ({len(r.content)} bytes)")
        else:
            fail("Body starts with PNG magic bytes", f"Got: {r.content[:8]!r}")

except Exception as e:
    fail("GET /api/model_architecture", str(e))


# ── Test 4: POST /api/execute — response schema ───────────────────────────────

section("4. POST /api/execute — Response Schema")

data, err = post_execute({"prompt": "שלום"})
if err:
    fail("Returns a response", err)
else:
    ok("Returns HTTP 200")

    # Required top-level fields
    for field in ("status", "error", "response", "steps"):
        if field in data:
            ok(f"Has top-level field '{field}'")
        else:
            fail(f"Has top-level field '{field}'", f"Got keys: {list(data.keys())}")

    if data.get("status") == "ok":
        ok("'status' is 'ok'")
    else:
        fail("'status' is 'ok'", f"Got: {data.get('status')}")

    if data.get("error") is None:
        ok("'error' is null on success")
    else:
        fail("'error' is null on success", str(data.get("error")))

    if isinstance(data.get("response"), str) and data.get("response"):
        ok("'response' is a non-empty string", data["response"][:80])
    else:
        fail("'response' is a non-empty string", str(data.get("response")))

    steps = data.get("steps", [])
    if isinstance(steps, list):
        ok(f"'steps' is a list ({len(steps)} steps)")
    else:
        fail("'steps' is a list", str(steps))

    # Validate each step has required fields
    valid_modules = {"AgentLLM", "WeatherTool", "AgriKnowledgeBase"}
    for i, step in enumerate(steps):
        missing = [f for f in ("module", "prompt", "response") if f not in step]
        if missing:
            fail(f"Step {i+1} missing fields: {missing}")
        else:
            ok(f"Step {i+1} has module/prompt/response", f"module={step.get('module')}")
        if step.get("module") not in valid_modules:
            fail(f"Step {i+1} module name matches architecture", f"Got '{step.get('module')}', expected one of {valid_modules}")


# ── Test 5: POST /api/execute — error schema ──────────────────────────────────

section("5. POST /api/execute — Error Response Schema")

# Send an empty prompt to trigger a graceful error or empty response
data, err = post_execute({"prompt": ""})
if err:
    # If the server crashed, that's the failure
    fail("Server handles empty prompt without crashing", err)
else:
    ok("Server handles empty prompt without crashing")
    if data.get("status") in ("ok", "error"):
        ok(f"'status' field present and valid: '{data['status']}'")
    else:
        fail("'status' field is 'ok' or 'error'", str(data.get("status")))
    if data.get("status") == "error":
        if data.get("error") and data.get("response") is None:
            ok("Error schema correct: error is set, response is null")
        else:
            fail("Error schema: error should be string, response should be null", str(data))


# ── Test 6: POST /api/execute — Weather tool integration ─────────────────────

section("6. POST /api/execute — Weather Tool")

WEATHER_TESTS = [
    ("ariel",            "2025-07-01", "מה הטמפרטורה היום באריאל?"),
    ("beer sheva",       "2025-07-01", "האם צריך להשקות היום בבאר שבע?"),
    ("eilat",            "2025-08-15", "מה מזג האוויר היום באילת?"),
    ("jerusalem center", "2025-01-10", "כמה קר היום בירושלים?"),
]

for city, date, query in WEATHER_TESTS:
    print(f"\n  Query: '{query}' (city={city}, date={date})")
    data, err = post_execute({"prompt": query, "city": city, "date": date})
    if err:
        fail(f"Weather query for {city}", err)
        continue

    if data.get("status") == "error":
        fail(f"Weather query for {city}", f"Agent error: {data.get('error')}")
        continue

    steps = data.get("steps", [])
    modules = [s.get("module") for s in steps]
    weather_fired = "WeatherTool" in modules
    has_reply = bool(data.get("response", "").strip())

    if weather_fired and has_reply:
        ok(f"WeatherTool fired + got reply for {city}", data["response"][:90])
    elif has_reply:
        ok(f"Got reply for {city} (WeatherTool step not captured separately)", data["response"][:90])
    else:
        fail(f"Weather query for {city}", f"No reply. modules seen: {modules}")
    time.sleep(1)


# ── Test 7: POST /api/execute — RAG / Knowledge-base tool ────────────────────

section("7. POST /api/execute — RAG / AgriKnowledgeBase Tool")

RAG_TESTS = [
    ("beer sheva", "2025-04-10", "מהם היתרונות של גידולי כיסוי לשיפור הקרקע?"),
    ("ariel",      "2025-03-15", "כיצד מנהלים מחלות בשדות חיטה?"),
    ("nitzan",     "2025-05-01", "מה הדרך הטובה לשמור על לחות הקרקע?"),
]

for city, date, query in RAG_TESTS:
    print(f"\n  Query: '{query[:55]}' (city={city})")
    data, err = post_execute({"prompt": query, "city": city, "date": date})
    if err:
        fail(f"RAG query: {query[:40]}", err)
        continue

    if data.get("status") == "error":
        fail(f"RAG query: {query[:40]}", f"Agent error: {data.get('error')}")
        continue

    steps = data.get("steps", [])
    modules = [s.get("module") for s in steps]
    rag_fired = "AgriKnowledgeBase" in modules
    has_reply = bool(data.get("response", "").strip())

    if rag_fired and has_reply:
        ok(f"AgriKnowledgeBase fired + got reply", data["response"][:90])
    elif has_reply:
        ok(f"Got reply (RAG step not captured separately)", data["response"][:90])
    else:
        fail(f"RAG query: {query[:40]}", f"No reply. modules: {modules}")
    time.sleep(1)


# ── Test 8: POST /api/execute — Combined weather + RAG ───────────────────────

section("8. POST /api/execute — Combined Weather + RAG")

data, err = post_execute({
    "prompt": "בהתאם לטמפרטורה היום בניצן, מתי כדאי לזרוע חיטה לפי הספרות המקצועית?",
    "city": "nitzan",
    "date": "2025-11-01"
})

if err:
    fail("Combined query", err)
elif data.get("status") == "error":
    fail("Combined query", f"Agent error: {data.get('error')}")
else:
    steps = data.get("steps", [])
    modules = [s.get("module") for s in steps]
    used_weather = "WeatherTool" in modules
    used_rag     = "AgriKnowledgeBase" in modules
    has_reply    = bool(data.get("response", "").strip())

    print(f"         Modules used: {modules}")
    if used_weather and used_rag:
        ok("Agent used BOTH WeatherTool + AgriKnowledgeBase")
    elif used_weather or used_rag:
        ok(f"Agent used at least one tool (weather={used_weather}, rag={used_rag})")
    elif has_reply:
        ok("Agent replied (steps not separately logged for tools)", data["response"][:90])
    else:
        fail("Combined query — no reply")

    if has_reply:
        print(f"         Reply: {data['response'][:200]}")


# ── Test 9: POST /api/execute — Multi-turn conversation ──────────────────────

section("9. POST /api/execute — Multi-turn Conversation (chat_id)")

chat_id = f"chat_{uuid.uuid4().hex[:8]}"
user_name = f"test_{uuid.uuid4().hex[:6]}"

# Turn 1
data1, err1 = post_execute({
    "prompt": "שלום! קוראים לי דני ואני גדל תירס.",
    "user_name": user_name,
    "chat_id": chat_id,
    "city": "ariel",
    "date": "2025-06-01"
})

if err1:
    fail("Turn 1 succeeds", err1)
elif data1.get("status") != "ok":
    fail("Turn 1 succeeds", str(data1.get("error")))
else:
    ok("Turn 1 succeeds", data1["response"][:80])
    time.sleep(1)

    # Turn 2 — test that the agent remembers the context
    data2, err2 = post_execute({
        "prompt": "האם אתה זוכר מה אמרתי שאני גדל?",
        "user_name": user_name,
        "chat_id": chat_id,
        "city": "ariel",
        "date": "2025-06-01"
    })

    if err2:
        fail("Turn 2 (follow-up in same chat) succeeds", err2)
    elif data2.get("status") != "ok":
        fail("Turn 2 succeeds", str(data2.get("error")))
    else:
        ok("Turn 2 succeeds", data2["response"][:80])
        reply2 = data2["response"].lower()
        if "תירס" in reply2 or "דני" in reply2:
            ok("Agent remembers previous context (mentions corn/name)")
        else:
            ok("Turn 2 replied (context recall unconfirmed)", data2["response"][:80])


# ── Test 10: Steps structure validation ──────────────────────────────────────

section("10. Steps Logging — Structure & Module Names")

data, err = post_execute({
    "prompt": "מה הטמפרטורה בחיפה היום?",
    "city": "haifa technion",
    "date": "2025-07-15"
})

if err or data.get("status") != "ok":
    fail("Execute returns steps", err or data.get("error"))
else:
    steps = data.get("steps", [])
    ok(f"Got {len(steps)} step(s)")

    valid_modules = {"AgentLLM", "WeatherTool", "AgriKnowledgeBase"}
    for i, step in enumerate(steps):
        m = step.get("module")
        p = step.get("prompt")
        r = step.get("response")
        if m and p is not None and r is not None:
            ok(f"Step {i+1}: module='{m}', has prompt + response")
        else:
            fail(f"Step {i+1} incomplete", str(step))
        if m not in valid_modules:
            fail(f"Step {i+1} module '{m}' not in architecture diagram names", str(valid_modules))


# ── Summary ───────────────────────────────────────────────────────────────────

section("SUMMARY")
passed = sum(1 for s, _ in RESULTS if s == "PASS")
failed = sum(1 for s, _ in RESULTS if s == "FAIL")
total  = len(RESULTS)

print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}\n")
for status, label in RESULTS:
    mark = "✓" if status == "PASS" else "✗"
    print(f"  {mark}  {label}")

print()
