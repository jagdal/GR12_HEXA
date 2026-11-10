"""Recalcule les scores des résultats LLM enregistrés (sans relancer les modèles),
après une modification des règles de notation de bench.py.

Usage : python benchmark/rescore.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import RESULTS, score  # noqa: E402
from questions import QUESTIONS  # noqa: E402

by_q = {it["q"]: it for it in QUESTIONS}
for f in sorted(RESULTS.glob("llm_*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for r in d["rows"]:
        r["ok"] = score(by_q[r["q"]], r["answer"])
    ans = [r for r in d["rows"] if r["answerable"]]
    una = [r for r in d["rows"] if not r["answerable"]]
    d["accuracy"] = round(sum(r["ok"] for r in ans) / len(ans), 3)
    d["refusal_rate"] = round(sum(r["ok"] for r in una) / len(una), 3)
    d["overall"] = round(sum(r["ok"] for r in d["rows"]) / len(d["rows"]), 3)
    f.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{d['model']:15} justesse {d['accuracy']:.0%}  refus {d['refusal_rate']:.0%}  global {d['overall']:.0%}")
