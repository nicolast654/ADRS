import subprocess
import re
import json
import os

CONFIG_FILE = "/home/nicolas/ChampSim/champsim_config.json"
CHAMPSIM_BIN = "/home/nicolas/ChampSim/bin/champsim"
TRACE = "/home/nicolas/ChampSim/Perlbench 41B ChampSim Trace.xz"

PREDICTORS = ["bimodal", "gshare", "perceptron", "hashed_perceptron"]

def run_champsim(bp_name):
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    config["ooo_cpu"][0]["branch_predictor"] = bp_name
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    subprocess.run(["./config.sh", CONFIG_FILE], check=True)
    subprocess.run(["make"], check=True)

    result = subprocess.run(
        [CHAMPSIM_BIN,
         "--warmup-instructions", "1000000",
         "--simulation-instructions", "5000000",
         TRACE],
        capture_output=True, text=True
    )

    out = result.stdout
    m = re.search(r"Branch Prediction Accuracy: .*?MPKI:\s+([\d.]+)", out)
    if not m:
        print(f"Couldn't find MPKI for {bp_name}")
        return None
    return float(m.group(1))

def main():
    results = {}
    for bp in PREDICTORS:
        print(f"Running {bp}...")
        mpki = run_champsim(bp)
        if mpki is not None:
            results[bp] = mpki

    print("\n=== Results ===")
    for bp, val in results.items():
        print(f"{bp}: {val:.3f} MPKI")

    best = min(results, key=results.get)
    print(f"\nBest predictor: {best} ({results[best]:.3f} MPKI)")

if __name__ == "__main__":
    main()

