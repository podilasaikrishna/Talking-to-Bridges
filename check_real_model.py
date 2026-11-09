"""Run the REAL trained model + healthy baseline on a real recording (no mocks).

From the project root:
    python check_real_model.py path/to/recording.xlsx     # raw Excel (.xlsx/.xlsm) or cleaned .csv
    python check_real_model.py                             # first recording in data/reference, if any
"""
import sys
from pathlib import Path

from analysis.assessment import assess_recording, load_predictor
from analysis.baseline import load_profile
from analysis.healthy_reference import _load_one, load_healthy_recordings, reference_dir

func, where = load_predictor()
print("model function:", where if func else f"NOT FOUND ({where})")
print("reference folder:", reference_dir(), "| exists:", reference_dir().is_dir())
print("healthy profile available:", load_profile() is not None)

if len(sys.argv) > 1:
    path = Path(sys.argv[1])
    df = _load_one(path)  # cleans raw Excel the same way the app does
    if df is None:
        sys.exit(f"Could not read {path} as a sensor recording (no Sensor_1..5 columns after cleaning).")
    print("recording:", path)
else:
    recs = load_healthy_recordings()
    if not recs:
        sys.exit("No recordings in the reference folder. Pass a file: python check_real_model.py path/to/file.xlsx")
    df = recs[0]
    print("recording: first healthy reference recording")

r = assess_recording(df)
print("status:", r["status"], "| model_available:", r["model_available"])
if r["model_error"]:
    print("model_error:", r["model_error"])
print("condition:", r["condition"], r["condition_confidence"])
print("severity_mm:", r["severity_mm"], r["severity_confidence"], "|", r["severity_label"])
b = r["baseline_comparison"]
print("baseline flag:", b.get("overall_flag"), "| agreement:", r["agreement"])
print("baseline note:", b.get("summary", ""))
for p in r["top_parameters"]:
    print("  -", p["source"], p["feature"], "| importance", p["importance"], "| change%", p["change_percent"])