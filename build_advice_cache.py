# -*- coding: utf-8 -*-
"""
build_advice_cache.py
=====================
Sabhi 38 PlantVillage disease classes ke liye advice pehle se generate
karke saved_models/advice_cache.json mein store karta hai.

Yeh sirf EK BAAR run karna hota hai.
Baad mein SmartAdvisor instant <2ms lookup use karta hai instead of
3449ms LLM API call — latency 88% se <1% tak girta hai.

Run:  venv\\Scripts\\python.exe build_advice_cache.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.consts  import MODEL_DIR
from src.advisor import SmartAdvisor

# ── All 38 classes ─────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
    class_names = json.load(f)

advisor  = SmartAdvisor()
cache    = {}
errors   = []
languages = ["English"]   # add "Hindi" etc. if needed

print(f"Building advice cache for {len(class_names)} classes...\n")
total_start = time.perf_counter()

for idx, disease in enumerate(class_names):
    cache[disease] = {}
    for lang in languages:
        t0 = time.perf_counter()
        try:
            advice = advisor.get_treatment_advice(disease, 90.0, language=lang)
            cache[disease][lang] = advice
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [{idx+1:>2}/{len(class_names)}] {disease[:50]:<50} [{lang}] {elapsed:.0f}ms")
        except Exception as e:
            errors.append((disease, lang, str(e)))
            print(f"  [ERROR] {disease} [{lang}]: {e}")

total_ms = (time.perf_counter() - total_start) * 1000

# Save cache
cache_path = os.path.join(MODEL_DIR, "advice_cache.json")
with open(cache_path, "w", encoding="utf-8") as f:
    json.dump(cache, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  Cache built: {len(cache)} diseases × {len(languages)} languages")
print(f"  Errors:      {len(errors)}")
print(f"  Total time:  {total_ms:.0f} ms  (one-time cost)")
print(f"  Saved to:    {cache_path}")
print(f"  Future lookups will take <2ms each (vs {total_ms/len(class_names):.0f}ms avg)")
