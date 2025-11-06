# openevolve/examples/ADRS/bp_optimizer/evaluator.py

import subprocess
import re
import json
from pathlib import Path

CHAMPSIM_ROOT = Path("/home/nicolas/ChampSim")

BIMODAL_CC = CHAMPSIM_ROOT / "branch" / "bimodal" / "bimodal.cc"
CHAMPSIM_BIN = CHAMPSIM_ROOT / "bin" / "champsim"
TRACE = CHAMPSIM_ROOT / "Perlbench 41B ChampSim Trace.xz"

def run_champsim() -> str:
    """
    Runs champsim and returns the output.
    """
    subprocess.run(["make", "clean", "-C", str(CHAMPSIM_ROOT)], check=True) # -C flag is for directory, check=True checks for error (return code != 0)
    subprocess.run(["make", "-C", str(CHAMPSIM_ROOT)], check=True) # -C flag is for directory, check=True checks for error (return code != 0)

    result = subprocess.run(
        [
            str(CHAMPSIM_BIN),
            "--warmup-instructions", "100000",
            "--simulation-instructions", "500000",
            str(TRACE)
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ChampSim failed")

    return result.stdout


def parse_mpki(output: str) -> float:
    """
    Parse the MPKI from ChampSim output line:
    CPU 0 Branch Prediction Accuracy: 94.45% MPKI: 11.39 ...
    """
    m = re.search(
        r"Branch Prediction Accuracy:\s*[\d.]+%.*?MPKI:\s*([\d.]+)",
        output,
    )
    if not m:
        raise ValueError("Could not find MPKI line in ChampSim output.")
    return float(m.group(1))


def evaluate(program_path: str) -> dict:
    """
    Main entry: ADRS will pass us a path to a C++ file (the LLM's candidate).
    We overwrite bimodal.cc with it, build, run, and return metrics.
    """
    program_path = Path(program_path)

    code = program_path.read_text()

    BIMODAL_CC.write_text(code)

    try:
        stdout = run_champsim()
        mpki = parse_mpki(stdout)

        return {
            "mpki": float(mpki),
            "objective": float(mpki),
            "success": 1.0,
        }
    except Exception as e:
        return {
            "mpki": 1e9,
            "objective": 1e9,
            "success": 0.0,
            "error": str(e),
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python evaluator.py path/to/candidate_bimodal.cc")
        raise SystemExit(1)

    metrics = evaluate(sys.argv[1])
    print(json.dumps(metrics, indent=2))

