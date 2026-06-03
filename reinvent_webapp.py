#!/usr/bin/env python3
"""
REINVENT4 + GNINA Web Application
=================================

A FastAPI web app for goal-directed de novo molecular design: REINVENT4 runs
staged-learning reinforcement learning where the reward is a GNINA docking score
(CNNaffinity / CNN pose score / Vina) against a user-prepared receptor.  The
agent learns, over many steps, to generate molecules predicted to bind the
target.

Architecture
------------
  * Serving env (`gnina_webapp`):  this FastAPI app + RDKit for image rendering.
  * `reinvent4` env:               runs the `reinvent` CLI (the RL loop).
  * `gnina-dock` env:              runs gnina_score.py (3D embed + GNINA docking),
                                   invoked by REINVENT's ExternalProcess component.
  * `openmmdl` env:                runs protprep.py for protein preparation.

The receptor-prep endpoints mirror the sibling GNINA docking app.

Run:
  conda run -n gnina_webapp uvicorn reinvent_webapp:app --host 0.0.0.0 --port 5012
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import (
    FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

# ── Paths / external tools ───────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(os.environ.get("REINVENT_WORK_DIR", "/tmp/reinvent_work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

GNINA_PATH = os.environ.get("GNINA_PATH", "/opt/gnina/gnina.1.3.2")
REINVENT_BIN = os.environ.get(
    "REINVENT_BIN", "/home/evehom/Programs/miniconda3/envs/reinvent4/bin/reinvent"
)
# The scorer env (reinvent4-web) has RDKit + CDPKit so gnina_score.py can run
# ligfilter/ligprep itself; cxcalc (ChemAxon) is expected on the system PATH.
SCORER_PYTHON = os.environ.get(
    "SCORER_PYTHON",
    "/home/evehom/Programs/miniconda3/envs/reinvent4-web/bin/python",
)
SCORE_SCRIPT = str(APP_DIR / "gnina_score.py")
LIGFILTER_SCRIPT = os.environ.get(
    "LIGFILTER_SCRIPT", "/opt/webapps/ligprepper/ligfilter.py"
)
LIGPREP_SCRIPT = os.environ.get(
    "LIGPREP_SCRIPT", "/opt/webapps/ligprepper/ligprep.py"
)
# Filter rule files travel WITH the webapp (copied into APP_DIR) so the app is
# self-contained and edits here take effect — ligfilter otherwise defaults to
# the copies in its own script directory. Passed as --*-file overrides below.
PAINS_FILE = APP_DIR / "PAINS.txt"
REOS_FILE = APP_DIR / "REOS.txt"
CUSTOM_FILTERS_FILE = APP_DIR / "custom_filters.txt"
PRIOR_FILE = os.environ.get(
    "REINVENT_PRIOR", "/home/evehom/Programs/REINVENT4/priors/reinvent.prior"
)
PROTPREP_SCRIPT = os.environ.get(
    "PROTPREP_SCRIPT", str(Path("/opt/webapps/gnina/protprep.py"))
)


def _find_openmmdl_python() -> str:
    """Locate the openmmdl env python used for protein preparation."""
    explicit = os.environ.get("OPENMMDL_PYTHON")
    if explicit and Path(explicit).exists():
        return explicit
    guess = "/home/evehom/Programs/miniconda3/envs/openmmdl/bin/python"
    if Path(guess).exists():
        return guess
    try:
        out = subprocess.run(
            ["conda", "run", "-n", "openmmdl", "python", "-c",
             "import sys; print(sys.executable)"],
            capture_output=True, text=True, timeout=30,
        )
        cand = out.stdout.strip()
        if cand and Path(cand).exists():
            return cand
    except Exception:
        pass
    return "python"


OPENMMDL_PYTHON = _find_openmmdl_python()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(APP_DIR / "reinvent_webapp.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("reinvent_webapp")

app = FastAPI(title="REINVENT4 + GNINA")
if (APP_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

# ── Job state ────────────────────────────────────────────────────────────────
LOG_LINE_RE = re.compile(
    r"Score:\s*([-\d.]+)\s*Agent NLL:\s*([-\d.]+)\s*"
    r"Valid:\s*(\d+)%\s*Step:\s*(\d+)"
)


@dataclass
class Job:
    job_id: str
    job_dir: Path
    name: str
    metric: str
    max_steps: int
    status: str = "starting"          # starting|running|completed|failed|stopped
    message: str = ""
    proc: Optional[asyncio.subprocess.Process] = None
    steps: List[Dict[str, float]] = field(default_factory=list)  # per-step summary
    best: List[Dict[str, Any]] = field(default_factory=list)     # top molecules
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None
    canon: bool = True                # canonicalize tautomers in grid + CSV export


JOBS: Dict[str, Job] = {}
WS_CLIENTS: Dict[str, List[WebSocket]] = {}

_TAUT_ENUM = None  # process-wide, lazily built


def canonical_smiles(smi: str) -> str:
    """Return the canonical-tautomer SMILES; fall back to the input on failure.

    Mirrors the scorer's embed-time canonicalization so the molecules shown in
    the grid / CSV match the dominant tautomer that was actually docked.
    """
    global _TAUT_ENUM
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
        if _TAUT_ENUM is None:
            _TAUT_ENUM = rdMolStandardize.TautomerEnumerator()
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi
        return Chem.MolToSmiles(_TAUT_ENUM.Canonicalize(m))
    except Exception:
        return smi


# ── Helpers ──────────────────────────────────────────────────────────────────
def secure_filename(filename: str) -> str:
    filename = os.path.basename(filename or "")
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return filename or "upload.dat"


async def broadcast(job_id: str, payload: dict) -> None:
    """Send a JSON payload to all websocket clients watching a job."""
    for ws in list(WS_CLIENTS.get(job_id, [])):
        try:
            await ws.send_json(payload)
        except Exception:
            try:
                WS_CLIENTS[job_id].remove(ws)
            except ValueError:
                pass


def job_snapshot(job: Job) -> dict:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "name": job.name,
        "metric": job.metric,
        "max_steps": job.max_steps,
        "steps": job.steps,
        "best": job.best,
    }


# ── Score-transform defaults per docking metric ──────────────────────────────
METRIC_DEFAULTS = {
    # CNNaffinity: predicted pK, higher = stronger binder.
    "cnn_affinity": {"transform": "sigmoid", "low": 4.0, "high": 8.0, "k": 0.5},
    # CNN pose score: already in [0, 1], higher = more confident pose.
    "cnn_pose_score": {"transform": "sigmoid", "low": 0.3, "high": 0.9, "k": 0.5},
    # Vina affinity: kcal/mol, more negative = stronger binder.
    "vina_affinity": {"transform": "reverse_sigmoid", "low": -12.0, "high": -5.0, "k": 0.5},
}


def build_ligfilter_flags(p: dict) -> List[str]:
    """Translate UI options into ligfilter.py CLI flags."""
    flags: List[str] = []
    # Use the webapp's own copies of the rule files (--*-file implies the filter).
    if p.get("filter_pains"):
        flags += ["--pains-file", str(PAINS_FILE)]
    if p.get("filter_reos"):
        flags += ["--reos-file", str(REOS_FILE)]
    if p.get("filter_custom", True):
        flags += ["--custom-file", str(CUSTOM_FILTERS_FILE)]
    if p.get("filter_ro5"):
        flags.append("--ro5")
    if p.get("filter_ro3"):
        flags.append("--ro3")
    for opt, flag in (("filter_mw", "--mw"), ("filter_logp", "--logp"),
                      ("filter_qed", "--qed"), ("filter_tpsa", "--tpsa"),
                      ("filter_hba", "--hba"), ("filter_hbd", "--hbd"),
                      ("filter_rb", "--rb"), ("filter_ha", "--ha")):
        val = str(p.get(opt, "")).strip()
        if val:
            flags += [flag, val]
    return flags


def build_ligprep_flags(p: dict) -> List[str]:
    """Translate UI options into ligprep.py CLI flags.

    ligprep is off by default (ETKDG is the standard 3D prep). When it *is*
    re-enabled, apply RL-safe caps so a pathological prior-sampled molecule
    can't stall a step: prior samples often carry many unspecified stereocentres,
    and ligprep's defaults (up to 32 isomers, each CONFORGE'd for 60 s) can run
    for many minutes. Cap enumeration low and shorten the CONFORGE budget; both
    stay overridable via params.
    """
    flags = ["--pH", str(p.get("ligprep_ph", 7.4)),
             "--mode", p.get("ligprep_mode", "dominant")]
    if str(p.get("ligprep_max_tautomers", "")).strip():
        flags += ["--max-tautomers", str(p["ligprep_max_tautomers"]).strip()]
    if p.get("ligprep_no_stereo"):
        flags.append("--no-stereo")
    else:
        flags += ["--max-stereo",
                  str(p.get("ligprep_max_stereo", "")).strip() or "4"]
    flags += ["--conforge-timeout",
              str(p.get("ligprep_conforge_timeout", "")).strip() or "20"]
    return flags


def build_score_config(job_dir: Path, p: dict) -> Path:
    """Write the gnina_score.py JSON config for this job."""
    metric = p.get("metric", "vina_affinity")
    # CNN scoring is synced to the reward: Vina reward needs no CNN (none =
    # fastest); a CNN reward requires the CNN to run, so never 'none'.
    cnn_scoring = p.get("cnn_scoring") or (
        "none" if metric == "vina_affinity" else "rescore")
    # CPU budget shared by docking / prep / filter (machine has 36 cores).
    cpu = int(p.get("dock_cpu", 32))
    cfg = {
        "gnina_path": GNINA_PATH,
        "receptor": str(job_dir / "receptor.pdb"),
        "autobox_ligand": str(job_dir / "ref_ligand.sdf"),
        "autobox_add": float(p.get("autobox_add", 4.0)),
        "exhaustiveness": int(p.get("exhaustiveness", 4)),
        "num_modes": 1,
        "cnn_scoring": cnn_scoring,
        "cpu": cpu,
        "gpu": p.get("docking_gpu", "0"),
        "property": metric,
        "work_dir": str(job_dir / "score_work"),
        "embed_workers": cpu,
        "timeout": int(p.get("dock_timeout", 1800)),
        # Vina-only docking runs one single-CPU GNINA process per state across
        # `dock_workers` cores; `dock_timeout` is the per-molecule budget.
        "parallel_dock": bool(p.get("parallel_dock", True)),
        "dock_workers": cpu,
        "dock_timeout": int(p.get("per_dock_timeout", 120)),
        # ── ligand filtering / preparation ahead of docking ──
        "scorer_python": SCORER_PYTHON,
        "ligfilter_script": LIGFILTER_SCRIPT,
        "ligprep_script": LIGPREP_SCRIPT,
        "use_ligfilter": bool(p.get("use_ligfilter", True)),
        "ligfilter_flags": build_ligfilter_flags(p),
        "filter_workers": cpu,
        "use_ligprep": bool(p.get("use_ligprep", False)),
        "ligprep_flags": build_ligprep_flags(p),
        "prep_workers": cpu,
        "prep_timeout": int(p.get("prep_timeout", 1800)),
        # Canonicalize tautomers before ETKDG embedding so docking scores the
        # dominant tautomer, not a minor form REINVENT happened to emit.
        "canonicalize_tautomers": bool(p.get("canonicalize_tautomers", True)),
        "apply_state_penalty": bool(p.get("apply_state_penalty", False)),
    }
    cnn_model = p.get("cnn_model", "").strip()
    if cnn_model:
        cfg["cnn"] = cnn_model
    out = job_dir / "score_config.json"
    out.write_text(json.dumps(cfg, indent=2))
    return out


def build_reinvent_config(job_dir: Path, p: dict, score_config: Path) -> Path:
    """Assemble the REINVENT4 staged_learning JSON config for this job."""
    metric = p.get("metric", "vina_affinity")
    defaults = METRIC_DEFAULTS.get(metric, METRIC_DEFAULTS["vina_affinity"])
    transform = {
        "type": p.get("transform_type", defaults["transform"]),
        "low": float(p.get("transform_low", defaults["low"])),
        "high": float(p.get("transform_high", defaults["high"])),
        "k": float(p.get("transform_k", defaults["k"])),
    }

    components: List[dict] = [
        {
            "ExternalProcess": {
                "endpoint": [
                    {
                        "name": f"gnina {metric}",
                        "weight": float(p.get("dock_weight", 1.0)),
                        "params": {
                            "executable": SCORER_PYTHON,
                            "args": f"{SCORE_SCRIPT} --config {score_config}",
                            "property": metric,
                        },
                        "transform": transform,
                    }
                ]
            }
        }
    ]

    if p.get("use_qed"):
        components.append({"QED": {"endpoint": [
            {"name": "QED", "weight": float(p.get("qed_weight", 0.3))}
        ]}})

    if p.get("use_mw"):
        mw_low, mw_high = float(p.get("mw_low", 200.0)), float(p.get("mw_high", 400.0))
        components.append({"MolecularWeight": {"endpoint": [{
            "name": "Molecular weight",
            "weight": float(p.get("mw_weight", 0.3)),
            "transform": {
                "type": "double_sigmoid",
                "low": mw_low, "high": mw_high,
                # coef_div centers the double_sigmoid on the window midpoint.
                "coef_div": (mw_low + mw_high) / 2, "coef_si": 15.0, "coef_se": 15.0,
            },
        }]}})

    if p.get("use_slogp"):
        sl_low, sl_high = float(p.get("slogp_low", 1.0)), float(p.get("slogp_high", 3.0))
        components.append({"SlogP": {"endpoint": [{
            "name": "SlogP",
            "weight": float(p.get("slogp_weight", 0.25)),
            "transform": {
                "type": "double_sigmoid",
                "low": sl_low, "high": sl_high,
                "coef_div": (sl_low + sl_high) / 2, "coef_si": 2.0, "coef_se": 2.0,
            },
        }]}})

    config = {
        "run_type": "staged_learning",
        "device": p.get("train_device", "cuda:0"),
        "parameters": {
            "summary_csv_prefix": "results",
            "use_checkpoint": False,
            "purge_memories": False,
            "prior_file": PRIOR_FILE,
            "agent_file": PRIOR_FILE,
            "batch_size": int(p.get("batch_size", 64)),
            "unique_sequences": True,
            "randomize_smiles": True,
        },
        "learning_strategy": {
            "type": "dap",
            "sigma": float(p.get("sigma", 128)),
            "rate": float(p.get("learning_rate", 0.0001)),
        },
        "stage": [
            {
                "chkpt_file": str(job_dir / "agent.chkpt"),
                "termination": "simple",
                "max_score": float(p.get("max_score", 1.0)),
                "min_steps": int(p.get("min_steps", 5)),
                "max_steps": int(p.get("max_steps", 100)),
                "scoring": {"type": p.get("aggregation", "geometric_mean"),
                            "component": components},
            }
        ],
    }

    if p.get("use_diversity_filter", True):
        config["diversity_filter"] = {
            "type": p.get("df_type", "IdenticalMurckoScaffold"),
            "bucket_size": int(p.get("df_bucket", 25)),
            "minscore": float(p.get("df_minscore", 0.4)),
            "minsimilarity": float(p.get("df_minsim", 0.4)),
            "penalty_multiplier": float(p.get("df_penalty", 0.5)),
        }

    out = job_dir / "reinvent_config.json"
    out.write_text(json.dumps(config, indent=2))
    return out


def collect_best(job_dir: Path, metric: str, top_n: int = 24,
                 canonicalize: bool = True) -> List[Dict[str, Any]]:
    """Read the per-step results CSV and return the best unique molecules.

    Ranks by the raw docking metric (CNNaffinity/pose higher-better, Vina
    lower-better), keeping the best occurrence of each canonical SMILES.
    """
    import csv

    csv_path = job_dir / "results_1.csv"
    if not csv_path.exists():
        return []

    raw_col = {
        "cnn_affinity": "gnina cnn_affinity (raw)",
        "cnn_pose_score": "gnina cnn_pose_score (raw)",
        "vina_affinity": "gnina vina_affinity (raw)",
    }.get(metric)

    rows: Dict[str, Dict[str, Any]] = {}
    try:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            # The raw-metric column name follows REINVENT's "<endpoint> (raw)"
            # convention; fall back to any column containing the metric + raw.
            if raw_col not in headers:
                raw_col = next(
                    (h for h in headers if metric.split("_")[0] in h.lower()
                     and "raw" in h.lower()),
                    "Score",
                )
            pose_col = next((h for h in headers if "cnn_pose_score" in h), None)
            vina_col = next((h for h in headers if "vina_affinity" in h), None)
            qed_col = next((h for h in headers
                            if h.lower().startswith("qed") and "raw" in h.lower()), None)

            def fnum(val, default=0.0):
                # REINVENT writes 'None'/'' for endpoints not computed in this
                # run (e.g. CNN columns when cnn_scoring=none). Never let one such
                # cell raise — degrade to a default so the grid still populates.
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default

            for r in reader:
                smi = r.get("SMILES", "").strip()
                if not smi:
                    continue
                score = fnum(r.get("Score"))
                raw = fnum(r.get(raw_col))
                entry = {
                    "smiles": smi,
                    "score": round(score, 3),
                    "raw": round(raw, 3),
                    "pose": round(fnum(r.get(pose_col)), 3) if pose_col else None,
                    "vina": round(fnum(r.get(vina_col)), 3) if vina_col else None,
                    "qed": round(fnum(r.get(qed_col)), 3) if qed_col else None,
                    "step": int(fnum(r.get("step"))),
                }
                prev = rows.get(smi)
                if prev is None or entry["score"] > prev["score"]:
                    rows[smi] = entry
    except Exception as e:  # never let a malformed CSV crash the monitor
        logger.warning("collect_best failed: %s", e)
        return []

    reverse = metric != "vina_affinity"
    ranked = sorted(rows.values(), key=lambda e: e["raw"], reverse=reverse)[:top_n]
    if canonicalize:
        # Only the top_n displayed molecules — cheap even on the ~3 s poll.
        for e in ranked:
            e["smiles"] = canonical_smiles(e["smiles"])
    return ranked


async def monitor_job(job: Job, config_path: Path) -> None:
    """Launch REINVENT and stream progress to websocket clients."""
    log_path = job.job_dir / "run.log"
    cmd = [REINVENT_BIN, "-l", str(log_path), "-f", "json", str(config_path)]
    logger.info("Job %s: %s", job.job_id, " ".join(cmd))

    try:
        job.proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(job.job_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            # Lead a new session/process group so /stop can signal the whole
            # tree (reinvent → gnina_score.py → ligprep/cxcalc/gnina workers),
            # not just the reinvent parent. See _terminate_tree.
            start_new_session=True,
        )
    except Exception as e:
        job.status = "failed"
        job.message = f"Could not launch REINVENT: {e}"
        await broadcast(job.job_id, job_snapshot(job))
        return

    job.status = "running"
    job.message = "Reinforcement learning in progress…"
    await broadcast(job.job_id, job_snapshot(job))

    seen_steps = set()
    while True:
        done = job.proc.returncode is not None
        # Parse newly logged steps.
        if log_path.exists():
            try:
                text = log_path.read_text(errors="ignore")
            except Exception:
                text = ""
            for m in LOG_LINE_RE.finditer(text):
                score, nll, valid, step = m.groups()
                step = int(step)
                if step in seen_steps:
                    continue
                seen_steps.add(step)
                job.steps.append({
                    "step": step,
                    "score": round(float(score), 4),
                    "nll": round(float(nll), 2),
                    "valid": int(valid),
                })
            job.best = collect_best(job.job_dir, job.metric, canonicalize=job.canon)
            await broadcast(job.job_id, job_snapshot(job))

        if done:
            break
        try:
            await asyncio.wait_for(job.proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pass

    stderr = b""
    try:
        _, stderr = await job.proc.communicate()
    except Exception:
        pass

    job.finished = time.time()
    job.best = collect_best(job.job_dir, job.metric, canonicalize=job.canon)
    if job.status == "stopped":
        job.message = "Run stopped by user. Partial results available."
    elif job.proc.returncode == 0:
        job.status = "completed"
        job.message = f"Done. {len(job.steps)} steps completed."
    else:
        job.status = "failed"
        tail = (stderr or b"").decode(errors="ignore")[-1500:]
        job.message = f"REINVENT exited with code {job.proc.returncode}.\n{tail}"
    await broadcast(job.job_id, job_snapshot(job))
    logger.info("Job %s finished: %s", job.job_id, job.status)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    index = APP_DIR / "templates" / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>REINVENT4 + GNINA</h1><p>templates/index.html missing</p>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gnina": Path(GNINA_PATH).exists(),
        "reinvent": Path(REINVENT_BIN).exists(),
        "scorer_python": Path(SCORER_PYTHON).exists(),
        "prior": Path(PRIOR_FILE).exists(),
        "protprep": Path(PROTPREP_SCRIPT).exists(),
        "ligfilter": Path(LIGFILTER_SCRIPT).exists(),
        "ligprep": Path(LIGPREP_SCRIPT).exists(),
        "cxcalc": shutil.which("cxcalc") is not None,
        "openmmdl": OPENMMDL_PYTHON,
        "jobs": len(JOBS),
    }


@app.get("/molimage")
async def molimage(smiles: str, w: int = 300, h: int = 220):
    """Render a 2D PNG for a SMILES (used lazily by the results table)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(400, "invalid SMILES")
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    drawer.drawOptions().clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return Response(content=drawer.GetDrawingText(), media_type="image/png")


@app.post("/run")
async def run(
    name: str = Form("reinvent_gnina"),
    receptor_pdb_b64: str = Form(...),
    ref_ligand_sdf_b64: str = Form(...),
    params_json: str = Form("{}"),
):
    """Start a goal-directed RL job. Receptor + reference ligand come as base64
    (produced by the prep step or an upload), with all knobs in params_json."""
    try:
        p = json.loads(params_json)
    except json.JSONDecodeError:
        raise HTTPException(400, "params_json is not valid JSON")

    job_id = uuid.uuid4().hex[:10]
    job_dir = WORK_DIR / f"job_{job_id}"
    (job_dir / "score_work").mkdir(parents=True, exist_ok=True)

    try:
        (job_dir / "receptor.pdb").write_bytes(base64.b64decode(receptor_pdb_b64))
        (job_dir / "ref_ligand.sdf").write_bytes(base64.b64decode(ref_ligand_sdf_b64))
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, f"Could not decode receptor/ligand: {e}")

    metric = p.get("metric", "vina_affinity")
    score_cfg = build_score_config(job_dir, p)
    config_path = build_reinvent_config(job_dir, p, score_cfg)

    job = Job(
        job_id=job_id, job_dir=job_dir, name=secure_filename(name) or "run",
        metric=metric, max_steps=int(p.get("max_steps", 100)),
        canon=bool(p.get("canonicalize_tautomers", True)),
    )
    JOBS[job_id] = job
    asyncio.create_task(monitor_job(job, config_path))
    return {"job_id": job_id}


async def _terminate_tree(proc, grace: float = 10.0) -> None:
    """Stop a job's whole process tree, not just the reinvent parent.

    REINVENT is launched with ``start_new_session=True`` so it leads a process
    group containing the scorer (``gnina_score.py``) and its ligprep/cxcalc/gnina
    workers. SIGTERM the group, give it ``grace`` seconds to exit, then SIGKILL
    any stragglers. Avoids the orphaned-worker leak from a bare ``terminate()``.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
    except asyncio.TimeoutError:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@app.post("/stop/{job_id}")
async def stop(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.proc and job.proc.returncode is None:
        job.status = "stopped"
        await _terminate_tree(job.proc)
    return {"ok": True}


@app.get("/download/{job_id}/{kind}")
async def download(job_id: str, kind: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    files = {
        "csv": ("results_1.csv", "text/csv"),
        "chkpt": ("agent.chkpt", "application/octet-stream"),
        "log": ("run.log", "text/plain"),
        "config": ("reinvent_config.json", "application/json"),
    }
    if kind not in files:
        raise HTTPException(404, "unknown artifact")
    fname, media = files[kind]
    path = job.job_dir / fname
    if not path.exists():
        raise HTTPException(404, f"{fname} not available yet")

    # For the results CSV, add a Canonical_SMILES column (dominant tautomer)
    # next to REINVENT's raw SMILES, so the export matches what was docked while
    # preserving the original generated form. The raw column is left untouched.
    if kind == "csv" and job.canon:
        import csv as _csv
        from io import StringIO
        cache: Dict[str, str] = {}
        buf = StringIO()
        with open(path, newline="") as fh:
            reader = _csv.reader(fh)
            rows = list(reader)
        if rows:
            header = rows[0]
            try:
                si = header.index("SMILES")
            except ValueError:
                si = None
            writer = _csv.writer(buf)
            if si is None:
                writer.writerows(rows)
            else:
                writer.writerow(header[:si + 1] + ["Canonical_SMILES"] + header[si + 1:])
                for r in rows[1:]:
                    smi = r[si] if si < len(r) else ""
                    canon = cache.get(smi)
                    if canon is None:
                        canon = canonical_smiles(smi) if smi else ""
                        cache[smi] = canon
                    writer.writerow(r[:si + 1] + [canon] + r[si + 1:])
        return Response(content=buf.getvalue(), media_type=media,
                        headers={"Content-Disposition":
                                 f'attachment; filename="{job.name}_{fname}"'})

    return FileResponse(str(path), media_type=media,
                        filename=f"{job.name}_{fname}")


@app.websocket("/ws/{job_id}")
async def ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    WS_CLIENTS.setdefault(job_id, []).append(websocket)
    job = JOBS.get(job_id)
    if job:
        await websocket.send_json(job_snapshot(job))
    try:
        while True:
            # The browser only receives broadcasts; it never sends text. Use the
            # receive timeout as a keepalive heartbeat, NOT a disconnect — a long
            # run (hours) must keep streaming to an idle viewer. A real client
            # disconnect raises WebSocketDisconnect and breaks the loop.
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"keepalive": True})
                except Exception:
                    break  # peer gone — stop streaming to it
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        try:
            WS_CLIENTS.get(job_id, []).remove(websocket)
        except ValueError:
            pass


# ── Protein preparation (mirrors the GNINA docking app) ──────────────────────
@app.post("/protprep/inspect")
async def protprep_inspect(
    pdb_file: Optional[UploadFile] = File(None),
    pdb_id: Optional[str] = Form(None),
):
    """Load a PDB (upload or RCSB fetch) and report chains / HETATM groups so
    the user can choose chains, cofactors, and a reference ligand."""
    if not pdb_file and not (pdb_id and pdb_id.strip()):
        raise HTTPException(400, "Provide pdb_file or pdb_id")

    token = uuid.uuid4().hex[:8]
    prep_dir = WORK_DIR / f"protprep_{token}"
    prep_dir.mkdir(parents=True, exist_ok=True)

    try:
        if pdb_id and pdb_id.strip():
            pdb_id = pdb_id.strip().upper()
            pdb_path = prep_dir / f"{pdb_id}.pdb"
            fetch_script = prep_dir / "_fetch.py"
            fetch_script.write_text(
                "import urllib.request, urllib.error\n"
                "from pathlib import Path\n"
                f"out = Path({repr(str(pdb_path))})\n"
                f"pdb_id = {repr(pdb_id)}\n"
                "try:\n"
                "    urllib.request.urlretrieve(\n"
                "        f'https://files.rcsb.org/download/{pdb_id}.pdb', str(out))\n"
                "except urllib.error.HTTPError as e:\n"
                "    if e.code != 404:\n"
                "        raise\n"
                "    cif_tmp = out.with_suffix('.cif')\n"
                "    urllib.request.urlretrieve(\n"
                "        f'https://files.rcsb.org/download/{pdb_id}.cif', str(cif_tmp))\n"
                "    from pdbfixer import PDBFixer\n"
                "    from openmm.app import PDBFile\n"
                "    fixer = PDBFixer(filename=str(cif_tmp))\n"
                "    with open(str(out), 'w') as f:\n"
                "        PDBFile.writeFile(fixer.topology, fixer.positions, f)\n"
                "    cif_tmp.unlink()\n"
            )
            proc = await asyncio.create_subprocess_exec(
                OPENMMDL_PYTHON, str(fetch_script),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 or not pdb_path.exists():
                raise HTTPException(400, f"Could not fetch {pdb_id}: {stderr.decode()[:300]}")
        else:
            content = await pdb_file.read()
            pdb_path = prep_dir / secure_filename(pdb_file.filename)
            pdb_path.write_bytes(content)

        inspect_script = prep_dir / "_inspect.py"
        inspect_script.write_text(
            "import sys, json\n"
            f"sys.path.insert(0, {repr(str(Path(PROTPREP_SCRIPT).parent))})\n"
            "import protprep\n"
            "_noop = lambda *a, **k: None\n"
            "for _fn in ['_print','_ok','_warn','_info','_err','_step','_header','_rule']:\n"
            "    setattr(protprep, _fn, _noop)\n"
            "protprep._fatal = lambda msg: (_ for _ in ()).throw(SystemExit(msg))\n"
            "from pathlib import Path\n"
            f"info = protprep._inspect(Path({repr(str(pdb_path))}))\n"
            "print(json.dumps(info))\n"
        )
        proc = await asyncio.create_subprocess_exec(
            OPENMMDL_PYTHON, str(inspect_script),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise HTTPException(500, f"Inspect failed: {stderr.decode()[:500]}")
        info = json.loads(stdout.decode())
        return {
            "token": token,
            "pdb_filename": pdb_path.name,
            "chains": info["chains"],
            "chain_info": info["chain_info"],
            "n_std": info["n_std"],
            "n_water": info["n_water"],
            "het_groups": info["het_groups"],
            "n_altloc": info["n_altloc"],
            "ssbonds": len(info["ssbonds"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(prep_dir, ignore_errors=True)
        raise HTTPException(500, str(e))


@app.post("/protprep/run")
async def protprep_run(
    token: str = Form(...),
    keep_het: Optional[str] = Form(None),
    ph: float = Form(7.4),
    chains: Optional[str] = Form(None),
    cofactors: Optional[str] = Form(None),
):
    """Run the full prep pipeline; return prepared receptor + reference ligand
    as base64 for direct use in a docking-scored RL run."""
    prep_dir = WORK_DIR / f"protprep_{token}"
    if not prep_dir.exists():
        raise HTTPException(404, "Session not found — inspect the protein again")

    pdb_files = [f for f in prep_dir.glob("*.pdb") if not f.stem.startswith("_")]
    if not pdb_files:
        raise HTTPException(404, "PDB file not found in session")
    input_pdb = pdb_files[0]
    stem = input_pdb.stem
    output_pdb = prep_dir / f"{stem}_prepared.pdb"

    cmd = [
        OPENMMDL_PYTHON, PROTPREP_SCRIPT,
        "--input", str(input_pdb),
        "--output", str(output_pdb),
        "--ph", str(ph),
        "--no-pdb2pqr",
        "--minimize",
    ]
    if keep_het and keep_het.strip():
        cmd += ["--keep-het", keep_het]
    if chains and chains.strip():
        cmd += ["--chain"] + chains.split()
    if cofactors and cofactors.strip():
        cmd += ["--cofactor"] + cofactors.split()

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(prep_dir),
    )
    stdout, stderr = await proc.communicate()
    log_output = stdout.decode() + "\n" + stderr.decode()
    if proc.returncode != 0 or not output_pdb.exists():
        raise HTTPException(500, f"Protein preparation failed:\n{log_output[-2000:]}")

    minimized_pdb = prep_dir / f"{stem}_minimized.pdb"
    final_pdb = minimized_pdb if minimized_pdb.exists() else output_pdb
    lig_sdf = prep_dir / f"{stem}_prepared_ligand.sdf"

    return {
        "prepared_pdb_name": final_pdb.name,
        "prepared_pdb_b64": base64.b64encode(final_pdb.read_bytes()).decode(),
        "ligand_sdf_name": lig_sdf.name if lig_sdf.exists() else None,
        "ligand_sdf_b64": base64.b64encode(lig_sdf.read_bytes()).decode()
        if lig_sdf.exists() else None,
        "log": log_output[-3000:],
    }


if __name__ == "__main__":
    uvicorn.run("reinvent_webapp:app", host="0.0.0.0", port=5012,
                workers=1, reload=False, log_level="info")
