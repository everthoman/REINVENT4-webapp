# REINVENT4 + GNINA — Goal-directed Molecular Design Web App

A FastAPI web app for **structure-based de novo design**. [REINVENT4](https://github.com/MolecularAI/REINVENT4)
runs a *staged-learning* reinforcement-learning loop in which the reward for each
generated molecule is a [GNINA](https://github.com/gnina/gnina) docking score
against a target receptor. Over many steps the generative agent learns to
produce molecules predicted to bind the pocket.

```
 ┌─ REINVENT4 (reinvent4 env) ─ samples a batch of SMILES each step
 │        │
 │        ▼  ExternalProcess scoring component
 │   gnina_score.py (reinvent4-web env):
 │        ligfilter  → drop PAINS/REOS/property outliers (reward 0)
 │        ligprep    → salt-strip, tautomer/protomer @pH (cxcalc),
 │                     stereo, CONFORGE 3D states        [or RDKit ETKDG]
 │        GNINA dock → CNNaffinity / pose / Vina (best state per molecule)
 │        │
 │        ▼  sigmoid transform → reward in [0,1]
 └─ DAP policy update → repeat until max_steps / max_score
```

## Features

- **Goal-directed RL**: the agent is actively optimized toward better docking
  scores (not just generate-then-rank).
- **Ligand filtering + 3D prep before docking**: each generated batch passes
  through [`ligfilter`](../ligprepper/) (PAINS / REOS / Lipinski / property
  windows — rejects get reward 0 so the agent learns to avoid them), then a
  single **RDKit ETKDG** 3D conformer per molecule is embedded for docking — the
  default, fast and robust for an RL reward signal. Optionally enable
  [`ligprep`](../ligprepper/) (salt-strip → tautomer/protomer at pH via `cxcalc`
  → stereo enumeration → CONFORGE 3D) for chemically richer states instead;
  multiple states per molecule are all docked and the best score is kept
  (optionally penalized by `state_penalty`). When enabled, RL-safe caps apply
  (stereo enumeration ≤ 4, CONFORGE timeout 20 s) so a pathological molecule
  can't stall a step.
- **Reward metric choice**: **Vina affinity (default)**, GNINA `CNNaffinity`
  (predicted pK), or CNN pose score — each with a sensible score transform. The
  metric is **synced to GNINA's `cnn_scoring`**: a Vina reward runs GNINA
  Vina-only (`cnn_scoring=none`, fastest, CNN columns blank); a CNN reward forces
  `cnn_scoring=rescore` (the CNN must run, so `none` is disabled).
- **Integrated protein prep**: fetch a PDB by ID (or upload), pick chains /
  cofactors / reference ligand, then PDBFixer + OpenMM minimization. The
  reference ligand defines the GNINA autobox. *(Reuses the sibling GNINA docking
  app's `protprep.py`.)*
- **Upload path**: skip prep and upload a prepared receptor + reference ligand.
- **Live progress** over WebSocket: per-step mean score curve, % valid, best
  raw score, and a live grid of the top unique molecules with 2D structures.
- **Optional extra scoring**: QED, molecular-weight window, unwanted-substructure
  alerts, and a Murcko-scaffold diversity filter.
- **Downloads**: per-step results CSV, the trained agent checkpoint (`.chkpt`,
  reusable as a REINVENT `agent_file`), the run log, and the generated config.

## Quick start

```bash
./run.sh                      # serves on 0.0.0.0:5012
# or
conda run -n gnina_webapp uvicorn reinvent_webapp:app --host 0.0.0.0 --port 5012
```

Then open `http://<host>:5012`.

1. **Target receptor** — enter a PDB ID (e.g. `4N1U`), *Inspect*, choose the
   reference-ligand resname, *Prepare protein*. (Or switch to *Upload prepared*.)
2. **Optimization settings** — pick the reward metric, number of steps, batch
   size, etc.
3. **Start optimization** — watch the score climb and top molecules appear.

## Environments

The app orchestrates four existing conda envs (paths configurable via env vars):

| Env            | Role                                              | Used via |
|----------------|---------------------------------------------------|----------|
| `gnina_webapp` | serves this FastAPI app + RDKit images            | `run.sh` |
| `reinvent4`    | runs the `reinvent` RL CLI                        | `REINVENT_BIN` |
| `reinvent4-web`| runs `gnina_score.py` + `ligfilter`/`ligprep`     | `SCORER_PYTHON` |
| `openmmdl`     | runs `protprep.py` (protein prep)                 | `OPENMMDL_PYTHON` |

`reinvent4-web` (created for this app) has **RDKit + CDPKit**; `cxcalc`
(ChemAxon, for ligprep's tautomer/protomer step) must be on the system `PATH`
(`/usr/local/bin/cxcalc`).

```bash
mamba create -n reinvent4-web -c conda-forge python=3.11 rdkit tqdm numpy
conda run -n reinvent4-web pip install CDPKit==1.2.3
```

- GNINA binary: `GNINA_PATH` (default `/opt/gnina/gnina.1.3.2`).
- REINVENT prior: `REINVENT_PRIOR` (default the de-novo `reinvent.prior`).
- ligfilter/ligprep scripts: `LIGFILTER_SCRIPT` / `LIGPREP_SCRIPT`
  (default the sibling `/opt/webapps/ligprepper/` copies).

## Files

| File                 | Purpose |
|----------------------|---------|
| `reinvent_webapp.py` | FastAPI backend: prep endpoints, config generation, job monitor, WebSocket, downloads |
| `gnina_score.py`     | REINVENT4 `ExternalProcess` scorer: SMILES → 3D → GNINA → JSON payload |
| `templates/index.html` | Single-page UI |
| `run.sh`             | Launch script (port 5012) |

`gnina_score.py` can also be used standalone:

```bash
printf 'CCO\nc1ccccc1' | conda run -n reinvent4-web python gnina_score.py --config score_config.json
# → {"version":1,"payload":{"cnn_affinity":[...],"cnn_pose_score":[...],"vina_affinity":[...]}}
```

## Performance notes

GNINA docking dominates runtime, so how the batch is docked matters:

- **Vina-only (`cnn_scoring=none`, the default)** is CPU-bound. GNINA docks a
  multi-ligand file *sequentially* (the per-ligand search threads are capped by
  `exhaustiveness`), so a single big call leaves most cores idle. The scorer
  therefore docks **one single-CPU GNINA process per state, `dock_workers` at a
  time** (default = `dock_cpu`, with a per-molecule `dock_timeout` of 120 s).
  This saturates the CPU and means one slow molecule can't stall — or zero — the
  whole RL step. `exhaustiveness` defaults to **4** (good speed/quality trade for
  a reward signal; raise to 8+ for a final high-quality dock).
- **CNN scoring (`rescore` / `all`)** is GPU-bound, so the batch stays a single
  GNINA call — one process feeding the GPU avoids many workers fighting over GPU
  memory. Expect very roughly ~10 s per molecule here.

Set `parallel_dock=false` to force the single-call path for Vina too. Other
levers: reduce batch size, cap ligprep states per molecule, or split
training/docking across the two RTX 5000 GPUs (defaults: train on `cuda:0`,
dock on GPU `1`).

## How GNINA is wired into REINVENT

The generated `reinvent_config.json` adds an `ExternalProcess` scoring component:

```json
{"ExternalProcess": {"endpoint": [{
  "name": "gnina cnn_affinity", "weight": 1.0,
  "params": {"executable": ".../reinvent4-web/bin/python",
             "args": ".../gnina_score.py --config .../score_config.json",
             "property": "cnn_affinity"},
  "transform": {"type": "sigmoid", "low": 4.0, "high": 8.0, "k": 0.5}
}]}}
```

REINVENT pipes one SMILES per line to `gnina_score.py`, which returns a JSON
payload of per-molecule scores. The non-selected metrics ride along as metadata
and appear as extra CSV columns.
