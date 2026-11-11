"""Build the Genie space payload from genie/genie_space.md and create (or update) the space.

    python scripts/build_genie_space.py            # writes genie/genie_space.json only
    python scripts/build_genie_space.py --create   # + creates the space with the Databricks CLI
    python scripts/build_genie_space.py --update <space_id>

The markdown file stays the single source: tables (section 1), text instructions (section 2),
example SQL (section 3) and sample questions (section 4).
"""
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "genie" / "genie_space.md"
OUT = ROOT / "genie" / "genie_space.json"
TITLE = "Benelux Airbnb analyst"
DESCRIPTION = ("Airbnb listings in Brussels (City A), Antwerp (City B) and Amsterdam (City C): "
               "city, room-type and neighbourhood metrics, and the price model's predictions and errors.")
WAREHOUSE = "Serverless Starter Warehouse"


def section(md, number):
    m = re.search(rf"^## {number}\..*?$(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1)


def lines(text):
    parts = text.strip("\n").split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def ids(n):
    return sorted(uuid.uuid4().hex for _ in range(n))


def build():
    md = MD.read_text(encoding="utf-8")
    tables = sorted(set(re.findall(r"`(workspace\.airbnb_benelux\.gold_\w+)`", section(md, 1))))
    text = re.search(r"```\n(.*?)```", section(md, 2), re.S).group(1)
    examples = re.findall(r"\*\*(.+?)\*\*.*?```sql\n(.*?)```", section(md, 3), re.S)
    questions = re.findall(r"^\* (.+)$", section(md, 4), re.M)

    ex_ids, q_ids = ids(len(examples)), ids(len(questions))
    return {
        "version": 2,
        "config": {"sample_questions": [{"id": i, "question": [q]} for i, q in zip(q_ids, questions)]},
        "data_sources": {"tables": [{"identifier": t} for t in tables]},
        "instructions": {
            "text_instructions": [{"id": uuid.uuid4().hex, "content": lines(text)}],
            "example_question_sqls": [{"id": i, "question": [q], "sql": lines(s.strip())}
                                      for i, (q, s) in zip(ex_ids, examples)],
        },
    }


def cli(*args):
    out = subprocess.run([shutil.which("databricks"), *args], capture_output=True, text=True)
    if out.returncode:
        sys.exit(out.stderr)
    return json.loads(out.stdout)


def warehouse_id():
    for w in cli("warehouses", "list", "-o", "json"):
        if w["name"] == WAREHOUSE:
            return w["id"]
    sys.exit(f"warehouse {WAREHOUSE!r} not found")


if __name__ == "__main__":
    space = build()
    OUT.write_text(json.dumps(space, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(space['data_sources']['tables'])} tables, "
          f"{len(space['instructions']['example_question_sqls'])} example SQL, "
          f"{len(space['config']['sample_questions'])} sample questions")
    body = {"title": TITLE, "description": DESCRIPTION, "serialized_space": json.dumps(space)}
    if "--create" in sys.argv:
        r = cli("api", "post", "/api/2.0/genie/spaces", "--json", json.dumps({**body, "warehouse_id": warehouse_id()}))
        print("created space", r["space_id"])
    elif "--update" in sys.argv:
        sid = sys.argv[sys.argv.index("--update") + 1]
        cli("api", "patch", f"/api/2.0/genie/spaces/{sid}", "--json", json.dumps(body))
        print("updated space", sid)
