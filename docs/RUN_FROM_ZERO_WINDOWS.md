# Run the project from zero — Windows 10/11 (PowerShell)

A stranger's clean machine, no `make`, no admin rights needed except for optional tools. Every
command below is PowerShell. Linux/macOS equivalents are at the end.

**Needs:** Windows 10/11 x64, 8 GB RAM (16 GB recommended), ~5 GB free disk, internet access
(or a copy of `online_retail_II.xlsx`). Full pipeline: ~15–25 minutes on a laptop.

## 1. Install the tools (once)

```powershell
winget install --id Git.Git -e                                  # Git for Windows
winget install --id Microsoft.VCRedist.2015+.x64 -e             # C++ runtime (LightGBM needs vcomp140.dll)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # uv (Python manager)
# open a NEW PowerShell window so that git and uv are on PATH, then check:
git --version; uv --version
```

Optional: **Docker Desktop** (only for the container step), **Google Chrome** (Microsoft Edge, which
every Windows has, is enough for the PDF reports). `make` is **not** needed: `scripts\run.py` is the
Windows task runner and the Makefile only delegates to it.

## 2. Get the code

```powershell
git config --global core.longpaths true          # MLflow artifact paths can exceed 260 chars
mkdir C:\dev -Force; cd C:\dev                    # a short path without spaces avoids surprises
git clone https://github.com/OlgaMorgunova/boreal-marche-capstone.git
cd boreal-marche-capstone
```

Line endings are handled by `.gitattributes` (LF in the repository); nothing to configure.

## 3. Create the Python 3.11 environment

**Option A — uv (recommended, exact tested versions):**

```powershell
uv python install 3.11
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
uv pip install --python .venv\Scripts\python.exe -e . --no-deps
```

**Option B — plain pip** (Python 3.11 from python.org or `winget install Python.Python.3.11`):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

`requirements-dev.txt` = runtime + training + reports + dev tools, pinned and cross-platform (Linux-only
packages carry `sys_platform == 'linux'` markers). `requirements.txt` = runtime/serving only.

Shortcut used below (no need to activate the venv):

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py --version                                   # Python 3.11.x
```

(If you prefer activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then
`.venv\Scripts\Activate.ps1`.)

## 4. Data

Nothing to do if you are online: the first stage downloads the UCI zip and extracts the workbook.
Offline, copy the workbook to `doc\online_retail_II.xlsx`. Either way the SHA-256 is verified; to
check a copy yourself:

```powershell
Get-FileHash doc\online_retail_II.xlsx -Algorithm SHA256
# expected: BCBE73B35F5B7BABF197FB0CB983A11F5D9FF929078D4AA53D171B1F2DF2E980
```

## 5. Run the whole pipeline

```powershell
& $py scripts\run.py --list        # stages and the order of "all"
& $py scripts\run.py all           # data -> etl -> features -> dq -> spotcheck -> tune -> train
                                   # -> evaluate -> leakage -> rules-sweep -> rules -> rules-eval
```

What to expect: `data` ~1 min (Excel parsing), `tune` ~4 min, `rules` ~4 min (FP-Growth), the rest
seconds. Outputs: `data\bronze|silver|gold\`, `artifacts\model.joblib`, `artifacts\rules.json`,
`reports\metrics\*.json`, `mlflow.db`.

**The test set is not re-scored by default.** The repository records that it has been evaluated
(`reports\metrics\test_evaluation.json`), so `evaluate` prints *"Test set already evaluated … keeping
the recorded result"* and continues — this is intentional (test protection). To reproduce the test
number on your machine, run `& $py scripts\run.py evaluate --force` and report that look.

Single stages work the same way, e.g. `& $py scripts\run.py train evaluate`.

## 6. Verify

```powershell
& $py scripts\run.py dq            # silver OK / gold OK
& $py scripts\run.py test          # 50+ tests: unit, API contract, golden prediction, smoke
& $py scripts\run.py cov           # coverage gate 70 %
& $py scripts\run.py lint typecheck
```

Key numbers you should see (from `reports\metrics\`): validation PR-AUC 0.8102 (champion XGBoost),
threshold 0.56, 7,512 rules, hold-out hit-rate@5 0.145. Tiny floating-point differences between
operating systems are possible; the golden-prediction test uses the committed model artifact.

## 7. Reports (PDF)

```powershell
& $py scripts\run.py reports       # figures + reports\977_D1..D3_*.md/.html/.pdf
```

Uses Chrome or Edge in headless mode. If neither is found:
`$env:CHROME_PATH = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"`.

## 8. API and MLflow UI

```powershell
& $py scripts\run.py api           # http://127.0.0.1:8000/docs  (Ctrl+C to stop)
```

In a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/recommend -Method Post -ContentType "application/json" `
    -Body '{"basket": ["22916", "22918"], "k": 3}'
& .\.venv\Scripts\python.exe scripts\run.py mlflow-ui     # http://127.0.0.1:5000
```

## 9. Docker (optional)

```powershell
docker compose -f docker/compose.yml up --build            # API on http://127.0.0.1:8000
```

## 10. Cloud hyper-parameter search (optional)

```powershell
& $py scripts\run.py hpo --provider local                  # Optuna on this PC, $0
& $py scripts\run.py hpo --provider aws --dry-run          # SageMaker definition only
& $py scripts\run.py hpo --provider azure --dry-run        # Azure ML definition only
```

Real cloud runs need the extras (`uv pip install --python .venv\Scripts\python.exe -e ".[cloud-aws]"`
or `".[cloud-azure]"`), credentials and a budget alarm first — see `docs\CLOUD_OPTIONS.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: ... lib_lightgbm.dll` / `vcomp140.dll` | install the VC++ runtime (step 1) |
| `UnicodeEncodeError` in the console | `scripts\run.py` sets `PYTHONUTF8=1`; for direct `python -m ...` calls run `$env:PYTHONUTF8 = "1"` |
| `Activate.ps1 cannot be loaded` | use `$py` as above, or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `SHA-256 mismatch` | the workbook is not the UCI original — delete `data\raw\online_retail_II.xlsx` and re-run `data` |
| `Test set already evaluated` with an error | you called the module directly; use `scripts\run.py evaluate` or add `--force` knowingly |
| MLflow `FileNotFoundError` on long paths | `git config --global core.longpaths true` and a short project path (`C:\dev\...`) |
| Port 8000/5000 in use | `& $py -m uvicorn boreal_marche.api:app --port 8010` |
| Very slow first run | antivirus scanning `.venv`; exclude the project folder or wait once |

## Linux / macOS (same steps)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/OlgaMorgunova/boreal-marche-capstone.git && cd boreal-marche-capstone
make setup-pinned          # uv venv + requirements-dev.txt + editable install
make all && make test && make reports
make api                   # http://127.0.0.1:8000/docs
```

(macOS: `brew install libomp` if LightGBM complains about `libomp.dylib`.)
