# 🤖 AI Change Log — SmartFarmingAI

This folder is the **single source of truth** for every AI-assisted change made to this project.
Every session, feature addition, bug fix, model update, or architectural decision is recorded here.

---

## 📁 Folder Structure

```
ai/
├── README.md                    ← You are here — overview & conventions
├── CHANGELOG.md                 ← Master chronological log of all changes
├── sessions/                    ← One file per AI work session
│   └── YYYY-MM-DD_<topic>.md
├── decisions/                   ← Architecture & design decision records (ADRs)
│   └── ADR-001_<topic>.md
├── models/                      ← Model version history & performance snapshots
│   └── model_history.md
└── prompts/                     ← Key prompts that produced significant results
    └── YYYY-MM-DD_<purpose>.md
```

---

## 📌 How to Use

| Scenario | Where to record |
|---|---|
| New feature / bug fix by AI | `sessions/` + update `CHANGELOG.md` |
| Major design or architecture choice | `decisions/` (ADR format) |
| Model retrained / hyperparams changed | `models/model_history.md` |
| Prompt that produced a key output | `prompts/` |

---

## 🔖 Change Types (use in CHANGELOG.md)

| Tag | Meaning |
|---|---|
| `[ADD]` | New file, feature, or module added |
| `[MOD]` | Existing file or logic modified |
| `[FIX]` | Bug fix |
| `[DEL]` | File or feature removed |
| `[MODEL]` | ML model retrained, exported, or tuned |
| `[INFRA]` | Setup, dependencies, environment changes |
| `[REFACTOR]` | Code restructured without behaviour change |
| `[DOCS]` | Documentation only |

---

> **Rule:** Every AI interaction that changes files in this repo MUST have an entry in `CHANGELOG.md`
> and a corresponding session file under `sessions/`.
