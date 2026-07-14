#!/usr/bin/env python3
"""
End-to-end database test for the RPS social cognition backend.

Uses FastAPI's TestClient against an in-memory SQLite database (never touches
rps.db) and verifies the full session flow:

  1. Health check
  2. Participant registration (PUT /api/participants/{id}/registration)
  3. Registration lookup (GET)
  4. Session creation (POST /api/sessions)
  5. Block-order rotation (GET /api/sessions/{id}/rotation)
       - 6 blocks, each with avatar_id / condition / level
       - All 6 (condition, level) combos present exactly once
       - First block level < 2
       - No consecutive blocks with the same level
       - Avatar IDs resolved from registration
  6. Trial posting (POST /api/sessions/{id}/trials)
       - level field saved correctly
       - condition saved correctly
  7. Trigger posting (POST /api/sessions/{id}/triggers)
  8. Database integrity — query tables directly and verify row counts and values
  9. Rotation reproducibility — same session → same block order on repeated calls
 10. Different sessions → different block orders

Run from the backend/ directory:
  python3 test_e2e.py
"""

import os
import sys

# ── Use a temp file database so the test never touches rps.db ────────────────
# (In-memory SQLite requires StaticPool for multi-connection setups; a temp
#  file is simpler and avoids that complexity entirely.)

TEST_DB = "/tmp/test_rps_e2e.db"
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)

os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"

from fastapi.testclient import TestClient  # noqa: E402 (import after env set)
from app.main import app                   # noqa: E402
from app.database import engine            # noqa: E402
from app.models import Base, Session, Trial, ParticipantRegistration  # noqa: E402
from sqlalchemy.orm import sessionmaker    # noqa: E402

# Create all tables (lifespan not guaranteed to fire in TestClient without context manager)
Base.metadata.create_all(bind=engine)

client = TestClient(app)
DBSession = sessionmaker(bind=engine)


# ── Helpers ───────────────────────────────────────────────────────────────────

_pass = 0
_fail = 0

def check(label: str, condition: bool, detail: str = ""):
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  [PASS] {label}")
    else:
        _fail += 1
        print(f"  [FAIL] {label}" + (f"  ← {detail}" if detail else ""))


# ── Test data ─────────────────────────────────────────────────────────────────

PARTICIPANT_ID    = "TEST001"
FRIENDLY_AVATAR   = "r1_f_quinn"
NEUTRAL_AVATAR    = "r1_m_alex"


# ── 1. Health check ───────────────────────────────────────────────────────────

print("\n" + "="*60)
print("1. Health check")
print("="*60)

r = client.get("/api/health")
check("GET /api/health returns 200", r.status_code == 200)
check("Health response is ok", r.json().get("status") == "ok")


# ── 2. Participant registration ───────────────────────────────────────────────

print("\n" + "="*60)
print("2. Participant registration")
print("="*60)

r = client.put(
    f"/api/participants/{PARTICIPANT_ID}/registration",
    json={"friendly_avatar_id": FRIENDLY_AVATAR, "neutral_avatar_id": NEUTRAL_AVATAR},
)
check("PUT registration returns 200", r.status_code == 200, r.text)
reg = r.json()
check("friendly_avatar_id stored correctly", reg["friendly_avatar_id"] == FRIENDLY_AVATAR)
check("neutral_avatar_id stored correctly",  reg["neutral_avatar_id"]  == NEUTRAL_AVATAR)
check("participant_id in response",          reg["participant_id"]      == PARTICIPANT_ID)


# ── 3. Registration lookup ────────────────────────────────────────────────────

print("\n" + "="*60)
print("3. Registration lookup")
print("="*60)

r = client.get(f"/api/participants/{PARTICIPANT_ID}/registration")
check("GET registration returns 200", r.status_code == 200, r.text)
check("Lookup returns same avatars",  r.json()["friendly_avatar_id"] == FRIENDLY_AVATAR)

r404 = client.get("/api/participants/UNKNOWN/registration")
check("Unknown participant returns 404", r404.status_code == 404)


# ── 4. Session creation ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("4. Session creation")
print("="*60)

r = client.post("/api/sessions", json={
    "participant_id": PARTICIPANT_ID,
    "mode": "behavioral",
})
check("POST /api/sessions returns 200", r.status_code == 200, r.text)
session = r.json()
SESSION_ID = session["session_id"]
check("session_id is an integer",    isinstance(SESSION_ID, int))
check("participant_id matches",      session["participant_id"] == PARTICIPANT_ID)
check("mode matches",                session["mode"] == "behavioral")

r2 = client.get(f"/api/sessions/{SESSION_ID}")
check("GET /api/sessions/{id} returns 200", r2.status_code == 200)
check("Session detail participant matches",  r2.json()["participant_id"] == PARTICIPANT_ID)


# ── 5. Block-order rotation ───────────────────────────────────────────────────

print("\n" + "="*60)
print("5. Block-order rotation")
print("="*60)

r = client.get(f"/api/sessions/{SESSION_ID}/rotation")
check("GET rotation returns 200", r.status_code == 200, r.text)

blocks = r.json()["blocks"]
check("Rotation has exactly 6 blocks", len(blocks) == 6,
      f"got {len(blocks)}")

# Every block must have avatar_id, condition, level
for i, b in enumerate(blocks):
    check(f"Block {i+1} has avatar_id",  "avatar_id"  in b)
    check(f"Block {i+1} has condition",  "condition"  in b)
    check(f"Block {i+1} has level",      "level"      in b)

# Avatar IDs come from registration
for i, b in enumerate(blocks):
    expected = FRIENDLY_AVATAR if b["condition"] == "friendly" else NEUTRAL_AVATAR
    check(f"Block {i+1} avatar resolved from registration",
          b["avatar_id"] == expected,
          f"condition={b['condition']} expected {expected} got {b['avatar_id']}")

# All 6 (condition, level) combinations appear exactly once
combos = [(b["condition"], b["level"]) for b in blocks]
expected_combos = {("friendly", l) for l in (0, 1, 2)} | {("neutral", l) for l in (0, 1, 2)}
check("All 6 (condition, level) combos present",
      set(combos) == expected_combos,
      f"got {sorted(combos)}")
check("No duplicate combos", len(combos) == len(set(combos)))

# Buergi constraints
first_level = blocks[0]["level"]
check("First block is not level 2",
      first_level < 2, f"first level = {first_level}")

consecutive_same = any(blocks[i]["level"] == blocks[i+1]["level"] for i in range(5))
check("No consecutive blocks share the same level", not consecutive_same,
      f"levels = {[b['level'] for b in blocks]}")

print(f"\n  Block order: " +
      " → ".join(f"{b['condition'][0].upper()}{b['level']}" for b in blocks))


# ── 6. Trial posting ──────────────────────────────────────────────────────────

print("\n" + "="*60)
print("6. Trial posting")
print("="*60)

# Post one trial for each block (simulating first trial of each block)
posted_trials = []
for block_num, block in enumerate(blocks, start=1):
    trial_payload = {
        "block":              block_num,
        "agent":              block["avatar_id"],
        "trial_in_block":     1,
        "trial_global":       block_num,
        "participant_choice": 1,      # Rock
        "agent_choice":       2,      # Paper (participant loses)
        "outcome":            "lose",
        "points_delta":       -3,
        "points_cumulative":  100 - 3 * block_num,
        "rt_ms":              412.5,
        "onset_ms":           1000.0 * block_num,
        "feedback_onset_ms":  1000.0 * block_num + 4000,
        "iti_duration_ms":    2000.0,
        "block_onset_ms":     1000.0 * block_num - 100,
        "condition":          block["condition"],
        "level":              block["level"],
    }
    r = client.post(f"/api/sessions/{SESSION_ID}/trials", json=trial_payload)
    check(f"Trial block {block_num} (cond={block['condition']}, level={block['level']}) posts 200",
          r.status_code == 200, r.text)
    posted_trials.append(r.json())

# Verify level and condition are saved in responses
for t, block in zip(posted_trials, blocks):
    check(f"Trial level={block['level']} round-trips correctly",
          t["level"] == block["level"], f"expected {block['level']} got {t.get('level')}")
    check(f"Trial condition={block['condition']} round-trips correctly",
          t["condition"] == block["condition"])


# ── 7. Trigger posting ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("7. Trigger posting")
print("="*60)

r = client.post(f"/api/sessions/{SESSION_ID}/triggers",
                json={"tr_number": 0, "t_ms": 0.0})
check("POST trigger returns 200", r.status_code == 200, r.text)
check("Trigger tr_number correct", r.json()["tr_number"] == 0)


# ── 8. Database integrity ─────────────────────────────────────────────────────

print("\n" + "="*60)
print("8. Database integrity (direct table queries)")
print("="*60)

db = DBSession()

# Session row
session_row = db.get(Session, SESSION_ID)
check("Session row exists in DB",           session_row is not None)
check("Session participant_id correct",     session_row.participant_id == PARTICIPANT_ID)
check("Session mode correct",               session_row.mode == "behavioral")

# Registration row
reg_row = db.get(ParticipantRegistration, PARTICIPANT_ID)
check("Registration row exists in DB",      reg_row is not None)
check("Friendly avatar in DB",              reg_row.friendly_avatar_id == FRIENDLY_AVATAR)
check("Neutral avatar in DB",               reg_row.neutral_avatar_id  == NEUTRAL_AVATAR)

# Trial rows
trials = db.query(Trial).filter(Trial.session_id == SESSION_ID).all()
check("6 trial rows in DB",                 len(trials) == 6, f"got {len(trials)}")
check("All trials have level != None",      all(t.level is not None for t in trials))
check("All trials have condition set",      all(t.condition in ("friendly","neutral") for t in trials))

# Level values match what was posted
db_levels = sorted(t.level for t in trials)
check("DB levels are [0,0,1,1,2,2]",        db_levels == [0, 0, 1, 1, 2, 2],
      f"got {db_levels}")

# Conditions match
db_conditions = sorted(t.condition for t in trials)
check("DB conditions are 3 friendly + 3 neutral",
      db_conditions == ["friendly","friendly","friendly","neutral","neutral","neutral"],
      f"got {db_conditions}")

db.close()


# ── 9. Rotation reproducibility ───────────────────────────────────────────────

print("\n" + "="*60)
print("9. Rotation reproducibility")
print("="*60)

r1 = client.get(f"/api/sessions/{SESSION_ID}/rotation").json()["blocks"]
r2 = client.get(f"/api/sessions/{SESSION_ID}/rotation").json()["blocks"]
check("Same session → same rotation on repeated calls",
      r1 == r2, f"\nCall 1: {r1}\nCall 2: {r2}")


# ── 10. Different sessions → different block orders ────────────────────────────

print("\n" + "="*60)
print("10. Different sessions → different block orders")
print("="*60)

# Create 5 more sessions for the same participant and collect their orders
orders = []
for _ in range(5):
    r = client.post("/api/sessions", json={
        "participant_id": PARTICIPANT_ID, "mode": "behavioral",
    })
    sid = r.json()["session_id"]
    rot  = client.get(f"/api/sessions/{sid}/rotation").json()["blocks"]
    orders.append(tuple((b["condition"], b["level"]) for b in rot))

unique_orders = len(set(orders))
check("≥2 distinct block orders across 5 new sessions",
      unique_orders >= 2, f"all 5 sessions got identical order: {orders[0]}")

# All must still satisfy the constraints
for i, (sid_blocks, order) in enumerate(zip(
        [client.get(f"/api/sessions/{j}/rotation").json()["blocks"] for j in range(SESSION_ID+1, SESSION_ID+6)],
        orders)):
    first_lvl = order[0][1]
    consec = any(order[j][1] == order[j+1][1] for j in range(5))
    check(f"Session {SESSION_ID+i+1}: first level < 2",       first_lvl < 2)
    check(f"Session {SESSION_ID+i+1}: no consecutive levels",  not consec)


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
total = _pass + _fail
print(f"SUMMARY: {_pass}/{total} checks passed")
if _fail == 0:
    print("All checks passed — backend is ready.")
else:
    print(f"WARNING: {_fail} check(s) failed — see above.")
print("="*60)

# Clean up temp database
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)

sys.exit(0 if _fail == 0 else 1)
