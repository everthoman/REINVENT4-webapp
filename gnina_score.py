#!/usr/bin/env python3
"""
gnina_score.py — REINVENT4 external scoring component backed by GNINA docking.

Designed to be called by REINVENT4's `ExternalProcess` scoring component during
goal-directed reinforcement learning (staged_learning).  REINVENT pipes one
SMILES per line on stdin; this script returns a JSON payload of the form

    {"version": 1, "payload": {"<property>": [...], "cnn_pose_score": [...], ...}}

with one value per input SMILES, in the *same order* as the input.

Pipeline per batch
------------------
  1. (optional) ligfilter — drop molecules with unwanted features (PAINS/REOS/
     custom SMARTS, property windows).  Rejected molecules keep their slot but
     receive a "bad" reward so the agent learns to avoid them.
  2. Build 3D structures for the survivors:
       * ligprep (default) — salt strip → tautomer/protomer at pH (cxcalc) →
         stereo enumeration → CONFORGE 3D (CDPKit).  May emit several states
         per molecule; all are docked and the best score is kept per input.
       * or RDKit ETKDG fallback (one conformer per molecule).
  3. Dock the whole batch in a single GNINA call against a fixed prepared
     receptor (autobox reference ligand or explicit box).
  4. Read the best pose per *input molecule* (grouping ligprep states by their
     input index) and report CNNaffinity / CNNscore / Vina, mapped back to the
     input order.

Molecules that are filtered out, fail preparation, or fail docking receive a
"bad" raw value so that REINVENT's score transform maps them to ~0.

ligfilter.py and ligprep.py are run with this script's own interpreter
(`sys.executable`), so this script must run in an env that has RDKit + CDPKit
(the `reinvent4-web` env); `cxcalc` (ChemAxon) must be on PATH for ligprep.
All knobs are read from a JSON config passed with `--config`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Raw values assigned to molecules that could not be scored.  Chosen so the
# usual sigmoid transforms (higher-is-better for affinity/score) map them to 0.
BAD_VALUE = {
    "cnn_affinity": 0.0,   # predicted pK, higher = better binder
    "cnn_pose_score": 0.0,  # CNN pose score in [0, 1]
    "vina_affinity": 0.0,   # Vina kcal/mol, more negative = better (0 ~ no binding)
}


# ── 3D embedding (RDKit fallback when ligprep is disabled) ───────────────────
# Whether to canonicalize the tautomer before embedding. Set by embed_with_rdkit
# before the worker pool forks, so each worker inherits the right value. REINVENT
# (ChEMBL prior) routinely emits minor tautomers — exocyclic-imine pyridines
# (N=c1cccc[nH]1), lactim heterocycles, etc. Docking the dominant tautomer
# instead avoids scoring/optimizing a representational artifact.
_CANON_TAUT = True
_TAUT_ENUM = None  # per-process lazily-built rdMolStandardize.TautomerEnumerator


def _canonical_taut_mol(mol):
    """Return the canonical-tautomer mol; fall back to the input on any failure."""
    global _TAUT_ENUM
    if _TAUT_ENUM is None:
        _TAUT_ENUM = rdMolStandardize.TautomerEnumerator()
    try:
        out = _TAUT_ENUM.Canonicalize(mol)
        return out if out is not None else mol
    except Exception:
        return mol


def _embed_one(args: Tuple[int, str]) -> Tuple[int, Optional[str]]:
    """Embed a single SMILES into a 3D molblock tagged with its index."""
    idx, smiles = args
    smiles = smiles.strip()
    if not smiles:
        return idx, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return idx, None
        if _CANON_TAUT:
            mol = _canonical_taut_mol(mol)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                return idx, None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
        except Exception:
            pass
        mol.SetProp("_Name", str(idx))
        return idx, Chem.MolToMolBlock(mol)
    except Exception:
        return idx, None


def embed_with_rdkit(indexed: List[Tuple[int, str]], workers: int,
                     out_sdf: Path, canonicalize: bool = True) -> None:
    """Embed (index, smiles) pairs and write a multi-ligand SDF named by index."""
    global _CANON_TAUT
    _CANON_TAUT = canonicalize  # set before the pool forks so workers inherit it
    blocks: Dict[int, str] = {}
    if workers <= 1:
        for item in indexed:
            idx, mb = _embed_one(item)
            if mb is not None:
                blocks[idx] = mb
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, mb in ex.map(_embed_one, indexed):
                if mb is not None:
                    blocks[idx] = mb
    writer = Chem.SDWriter(str(out_sdf))
    for idx in sorted(blocks):
        m = Chem.MolFromMolBlock(blocks[idx], sanitize=False)
        if m is not None:
            m.SetProp("_Name", str(idx))
            writer.write(m)
    writer.close()


# ── ligfilter / ligprep stages (sibling CLI scripts, run via sys.executable) ─
def write_indexed_smi(indexed: List[Tuple[int, str]], path: Path) -> None:
    """Write SMILES <tab> index, one per line (index used as the molecule name)."""
    with open(path, "w") as fh:
        for idx, smi in indexed:
            fh.write(f"{smi}\t{idx}\n")


def read_surviving_indices(smi_path: Path) -> Set[int]:
    """Read the index column from a ligfilter output .smi (col2 = name = index).

    ligfilter writes tab-separated ``SMILES<TAB>name``; the name is the original
    batch index we tagged in :func:`write_indexed_smi`.
    """
    survivors: Set[int] = set()
    if not smi_path.exists():
        return survivors
    for line in smi_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace("\t", " ").replace(",", " ").split() if p]
        if len(parts) >= 2:
            try:
                survivors.add(int(parts[-1]))
            except ValueError:
                continue
    return survivors


def run_ligfilter(cfg: dict, in_smi: Path, out_smi: Path) -> bool:
    """Run ligfilter on the batch. Returns True on success."""
    python = cfg.get("scorer_python") or sys.executable
    script = cfg["ligfilter_script"]
    cmd = [python, script, "-i", str(in_smi), "-o", str(out_smi),
           "--no-unique", "-j", str(cfg.get("filter_workers", 4))]
    cmd += list(cfg.get("ligfilter_flags", []))
    res = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=int(cfg.get("filter_timeout", 600)))
    if res.returncode != 0:
        print(f"ligfilter failed (rc={res.returncode}): {res.stderr[-800:]}",
              file=sys.stderr)
        return False
    return True


def run_ligprep(cfg: dict, in_smi: Path, out_sdf: Path) -> bool:
    """Run ligprep to build 3D states. Returns True on success."""
    python = cfg.get("scorer_python") or sys.executable
    script = cfg["ligprep_script"]
    cmd = [python, script, "-i", str(in_smi), "-o", str(out_sdf),
           "-j", str(cfg.get("prep_workers", 4))]
    cmd += list(cfg.get("ligprep_flags", []))
    env = dict(os.environ)
    # Ensure ChemAxon's cxcalc is reachable for the tautomer/protomer step.
    if "/usr/local/bin" not in env.get("PATH", ""):
        env["PATH"] = env.get("PATH", "") + ":/usr/local/bin"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env,
                         timeout=int(cfg.get("prep_timeout", 1800)))
    if res.returncode != 0 or not out_sdf.exists():
        print(f"ligprep failed (rc={res.returncode}): {res.stderr[-800:]}",
              file=sys.stderr)
        return False
    return True


# ── GNINA docking ────────────────────────────────────────────────────────────
def _gnina_cmd(cfg: dict, ligand_sdf: Path, out_sdf: Path,
               cpu: Optional[int] = None) -> List[str]:
    """Build a GNINA command line for one ligand file.

    ``cpu`` overrides ``cfg["cpu"]`` (the parallel path pins each worker to 1).
    """
    gnina = cfg.get("gnina_path", "/opt/gnina/gnina.1.3.2")
    cmd = [
        gnina,
        "-r", cfg["receptor"],
        "-l", str(ligand_sdf),
        "-o", str(out_sdf),
        "--num_modes", str(cfg.get("num_modes", 1)),
        "--exhaustiveness", str(cfg.get("exhaustiveness", 4)),
        "--cnn_scoring", cfg.get("cnn_scoring", "rescore"),
        "--seed", str(cfg.get("seed", 42)),
    ]
    cpu_val = cpu if cpu is not None else cfg.get("cpu")
    if cpu_val:
        cmd += ["--cpu", str(cpu_val)]
    if cfg.get("cnn"):
        cmd += ["--cnn", cfg["cnn"]]

    if cfg.get("autobox_ligand"):
        cmd += ["--autobox_ligand", cfg["autobox_ligand"],
                "--autobox_add", str(cfg.get("autobox_add", 4.0))]
    elif all(k in cfg for k in ("center_x", "center_y", "center_z")):
        cmd += ["--center_x", str(cfg["center_x"]),
                "--center_y", str(cfg["center_y"]),
                "--center_z", str(cfg["center_z"]),
                "--size_x", str(cfg.get("size_x", 22.5)),
                "--size_y", str(cfg.get("size_y", 22.5)),
                "--size_z", str(cfg.get("size_z", 22.5))]
    else:
        raise ValueError("config must provide autobox_ligand or center_{x,y,z}")
    return cmd


def _gnina_env(cfg: dict) -> dict:
    env = dict(os.environ)
    gpu = cfg.get("gpu")
    if gpu is not None and str(gpu) != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def run_gnina(cfg: dict, ligand_sdf: Path, out_sdf: Path) -> None:
    """Dock a multi-ligand SDF against the receptor in a single GNINA call.

    Used for GPU CNN scoring (``cnn_scoring`` != ``none``): one process feeds the
    GPU, so cross-molecule parallelism would only fight over GPU memory.
    """
    cmd = _gnina_cmd(cfg, ligand_sdf, out_sdf)
    subprocess.run(cmd, env=_gnina_env(cfg), timeout=int(cfg.get("timeout", 1800)),
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)


def _split_sdf(sdf: Path, out_dir: Path) -> List[Path]:
    """Split a multi-record SDF into one file per record, verbatim.

    Splitting on the ``$$$$`` delimiter (rather than via RDKit) preserves the
    molecule title (= input index) and every SD tag — ligprep's
    ``state_penalty`` / ``state_population`` etc. ride through to docking.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    block: List[str] = []
    for line in sdf.read_text().splitlines(keepends=True):
        block.append(line)
        if line.strip() == "$$$$":
            p = out_dir / f"state_{len(paths):04d}.sdf"
            p.write_text("".join(block))
            paths.append(p)
            block = []
    if "".join(block).strip():               # trailing record with no delimiter
        p = out_dir / f"state_{len(paths):04d}.sdf"
        p.write_text("".join(block))
        paths.append(p)
    return paths


def _dock_one_state(payload: Tuple[dict, str, str]) -> Optional[str]:
    """Dock a single-ligand SDF in its own GNINA process (module-level for Pool).

    Returns the docked-pose SDF path, or None if this state failed/timed out — a
    single bad molecule can no longer stall or zero the whole batch.
    """
    cfg, state_path, out_path = payload
    cmd = _gnina_cmd(cfg, Path(state_path), Path(out_path), cpu=1)
    try:
        subprocess.run(cmd, env=_gnina_env(cfg),
                       timeout=int(cfg.get("dock_timeout", 120)),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    return out_path if Path(out_path).exists() else None


def run_gnina_parallel(cfg: dict, ligand_sdf: Path, out_sdf: Path,
                       tmp: Path) -> None:
    """Dock each state in its own single-CPU GNINA process, N at a time.

    For Vina-only scoring (``cnn_scoring=none``) GNINA is CPU-bound and docks the
    ligands in a multi-ligand file *sequentially* (search threads within one
    ligand are capped by ``exhaustiveness``), leaving most cores idle.  Running
    one single-CPU process per state across ``dock_workers`` cores saturates the
    CPU instead.  Successful poses are concatenated into ``out_sdf`` so
    :func:`parse_results` is unchanged.
    """
    states = _split_sdf(ligand_sdf, tmp / "states")
    if not states:
        out_sdf.write_text("")
        return
    workers = int(cfg.get("dock_workers") or cfg.get("cpu") or 8)
    workers = max(1, min(workers, len(states)))
    docked_dir = tmp / "docked_states"
    docked_dir.mkdir(parents=True, exist_ok=True)
    payloads = [(cfg, str(sp), str(docked_dir / f"{sp.stem}_out.sdf"))
                for sp in states]

    outputs: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_dock_one_state, payloads):
            if res:
                outputs.append(res)

    with open(out_sdf, "w") as fh:           # concatenate into one multi-SDF
        for op in outputs:
            fh.write(Path(op).read_text())


def parse_results(out_sdf: Path, apply_state_penalty: bool) -> Dict[int, Dict[str, float]]:
    """Best pose per input index → {cnn_affinity, cnn_pose_score, vina}.

    Several ligprep states can share an input index (same _Name); the state with
    the highest CNNaffinity wins.  With ``apply_state_penalty`` the molecule's
    ``state_penalty`` (−log10 population) is subtracted from CNNaffinity before
    comparison, down-weighting unlikely tautomer/protomer states.
    """
    results: Dict[int, Dict[str, float]] = {}
    if not out_sdf.exists():
        return results
    for mol in Chem.SDMolSupplier(str(out_sdf), sanitize=False):
        if mol is None:
            continue
        try:
            idx = int(mol.GetProp("_Name"))
        except (KeyError, ValueError):
            continue

        def _getf(name: str, default: float = 0.0) -> float:
            try:
                return float(mol.GetProp(name))
            except (KeyError, ValueError):
                return default

        cnn_aff = _getf("CNNaffinity")
        rank_val = cnn_aff
        if apply_state_penalty and mol.HasProp("state_penalty"):
            rank_val = cnn_aff - _getf("state_penalty")
        rec = {
            "cnn_affinity": cnn_aff,
            "cnn_pose_score": _getf("CNNscore"),
            "vina_affinity": _getf("minimizedAffinity"),
            "_rank": rank_val,
        }
        prev = results.get(idx)
        if prev is None or rank_val > prev["_rank"]:
            results[idx] = rec
    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="GNINA docking scorer for REINVENT4")
    ap.add_argument("--config", required=True, help="JSON config file")
    ap.add_argument("--keep-temp", action="store_true", help="keep working dir")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    prop = cfg.get("property", "cnn_affinity")
    embed_workers = int(cfg.get("embed_workers", max(1, (os.cpu_count() or 4) // 2)))

    smiles_list = [line.rstrip("\n") for line in sys.stdin if line.strip() != ""]
    n = len(smiles_list)

    cnn_affinity = [BAD_VALUE["cnn_affinity"]] * n
    cnn_pose = [BAD_VALUE["cnn_pose_score"]] * n
    vina = [BAD_VALUE["vina_affinity"]] * n

    if n == 0:
        _emit(prop, cnn_affinity, cnn_pose, vina)
        return 0

    work_dir = Path(cfg.get("work_dir", tempfile.gettempdir()))
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="gnina_score_", dir=str(work_dir)))
    try:
        indexed = list(enumerate(smiles_list))
        in_smi = tmp / "input.smi"
        write_indexed_smi(indexed, in_smi)

        # 1. ligfilter — narrow to surviving indices (others stay "bad").
        surviving: Set[int] = set(range(n))
        active_smi = in_smi
        if cfg.get("use_ligfilter"):
            filt_smi = tmp / "filtered.smi"
            if run_ligfilter(cfg, in_smi, filt_smi):
                surviving = read_surviving_indices(filt_smi)
                active_smi = filt_smi
            # on failure: keep all indices, fall through with original SMILES

        if not surviving:
            _emit(prop, cnn_affinity, cnn_pose, vina)
            return 0

        # 2. Build 3D structures (ligprep, or RDKit fallback).
        ligand_sdf = tmp / "ligands.sdf"
        prepared = False
        if cfg.get("use_ligprep"):
            prep_sdf = tmp / "prepared.sdf"
            if run_ligprep(cfg, active_smi, prep_sdf):
                ligand_sdf = prep_sdf
                prepared = True
        if not prepared:
            # Embed only the surviving molecules with RDKit.
            keep = [(i, smiles_list[i]) for i in sorted(surviving)]
            embed_with_rdkit(keep, embed_workers, ligand_sdf,
                             canonicalize=bool(cfg.get("canonicalize_tautomers", True)))

        # 3. Dock the batch, then 4. read best pose per input index.
        #    Vina-only is CPU-bound → dock states in parallel single-CPU procs
        #    (robust to one slow molecule).  CNN scoring is GPU-bound → keep one
        #    process feeding the GPU.  ``parallel_dock=false`` forces monolithic.
        out_sdf = tmp / "docked.sdf"
        use_parallel = (cfg.get("cnn_scoring", "rescore") == "none"
                        and cfg.get("parallel_dock", True))
        try:
            if use_parallel:
                run_gnina_parallel(cfg, ligand_sdf, out_sdf, tmp)
            else:
                run_gnina(cfg, ligand_sdf, out_sdf)
            for idx, rec in parse_results(
                    out_sdf, bool(cfg.get("apply_state_penalty"))).items():
                if 0 <= idx < n:
                    cnn_affinity[idx] = rec["cnn_affinity"]
                    cnn_pose[idx] = rec["cnn_pose_score"]
                    vina[idx] = rec["vina_affinity"]
        except subprocess.CalledProcessError as e:
            print(f"gnina failed: {(e.stderr or b'').decode()[-1000:]}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print("gnina timed out", file=sys.stderr)
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)

    _emit(prop, cnn_affinity, cnn_pose, vina)
    return 0


def _emit(prop: str, cnn_affinity, cnn_pose, vina) -> None:
    payload = {
        "cnn_affinity": cnn_affinity,
        "cnn_pose_score": cnn_pose,
        "vina_affinity": vina,
    }
    if prop not in payload:
        payload[prop] = cnn_affinity
    print(json.dumps({"version": 1, "payload": payload}))


if __name__ == "__main__":
    sys.exit(main())
