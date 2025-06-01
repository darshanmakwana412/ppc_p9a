import re
import subprocess
import csv
from pathlib import Path

MC_values = list(range(32, 1024, 32))
# NC_values = list(range(64, 256, 8))
KC_values = list(range(16, 1024, 16))

template_path = Path("cp3b.cc")
template_code = template_path.read_text()

output_csv = Path("tuning_results.csv")
with output_csv.open("w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["MC", "NC", "KC", "GFLOPS"])

    for MC in MC_values:
        for KC in KC_values:

            code = template_code
            code = re.sub(r"constexpr int MC = \d+;", f"constexpr int MC = {MC};", code)
            code = re.sub(r"constexpr int NC = \d+;", f"constexpr int NC = {MC};", code)
            code = re.sub(r"constexpr int KC = \d+;", f"constexpr int KC = {KC};", code)

            tune_path = Path(f"tune_cp.cc")
            tune_path.write_text(code)

            result = subprocess.run(["./grading", "benchmark", "--remote", "--file", str(tune_path)], capture_output=True, text=True)
            m = re.search(r"Avg Achieved FLOP    : ([\d.]+) GFLOP/s", result.stdout)
            wall_time = float(m.group(1)) if m else float("inf")

            print(f"MC={MC}, NC={MC}, KC={KC} → {wall_time:.3f}s")
            writer.writerow([MC, MC, KC, wall_time])
