# notebook_src — how `AntrikshVashistha.ipynb` is built

The notebook is the deliverable and is **fully self-contained at runtime**: it imports nothing
from this directory and a reviewer never needs these files to run it.

They exist so the notebook can be *maintained*. Each module was written and tested standalone
before being assembled into cells, which is why the engine could be verified against the golden
table independently of the notebook plumbing.

## Rebuilding

```bash
python notebook_src/build_notebook.py          # writes ../AntrikshVashistha.ipynb
jupyter nbconvert --to notebook --execute --inplace AntrikshVashistha.ipynb
```

`build_notebook.py` splits each module on its `# ==== / # SECTION ...` banners, strips the
cross-module imports that exist only so the modules can be tested separately, and interleaves
the markdown. In the notebook everything shares one namespace.

## Layout

| File | Notebook sections |
|---|---|
| `core.py` | 3 policy registry · 4 claims · 6 deterministic engine |
| `core2.py` | 5 output contract · 7 tool layer · 8 agent loop · 9 reconciliation gate |
| `core3.py` | 8b model layer (LIVE / REPLAY / SIMULATED) · 10 orchestrator |
| `dash.py` | 12 dashboard · 12b review console |
| `evals.py` | 13 evaluation harness · 14 ablation |
| `build_notebook.py` | assembly + all prose |

## The one rule

**Edit the modules, never the notebook.** A change made directly in `AntrikshVashistha.ipynb`
is lost at the next rebuild. After editing, rebuild and re-execute so the committed notebook,
`outputs/` and the PNGs stay consistent with the source.
