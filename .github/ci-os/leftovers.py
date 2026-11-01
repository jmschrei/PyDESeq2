"""List pydeseq2-* folders in joblib's temporary folder (and the system's).

Fork-only CI helper. Prints one summary line starting with LEFTOVERS and
writes the same as JSON. Always exits 0: the job's result is the tests'.
"""

import argparse
import json
import os
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

record = {"label": args.label, "platform": sys.platform, "joblib_temp_dir": None, "error": None}
try:
	import joblib
	from joblib._memmapping_reducer import _get_temp_dir

	record["joblib_version"] = joblib.__version__
	record["joblib_temp_dir"], record["joblib_uses_shared_mem"] = _get_temp_dir("")
except Exception as error:  # noqa: BLE001
	record["error"] = "{}: {}".format(type(error).__name__, error)

candidates = [record["joblib_temp_dir"], tempfile.gettempdir(), "/dev/shm"]
candidates += [os.environ.get(k) for k in ("JOBLIB_TEMP_FOLDER", "TMPDIR", "TEMP", "TMP", "RUNNER_TEMP")]
searched, seen = [], set()
for folder in candidates:
	if not folder or not os.path.isdir(folder):
		continue
	real = os.path.realpath(folder)
	if real in seen:
		continue
	seen.add(real)
	searched.append(folder)

found = []
for folder in searched:
	try:
		names = sorted(os.listdir(folder))
	except OSError as error:
		record.setdefault("list_errors", []).append("{}: {}".format(folder, error))
		continue
	for name in names:
		if not name.startswith("pydeseq2-"):
			continue
		path = os.path.join(folder, name)
		files = []
		for root, _, filenames in os.walk(path):
			for filename in filenames:
				full = os.path.join(root, filename)
				try:
					size = os.path.getsize(full)
				except OSError:
					size = None
				files.append({"path": os.path.relpath(full, path), "bytes": size})
		found.append({"folder": path, "files": files})

record["searched"] = searched
record["found"] = found
with open(args.out, "w") as handle:
	json.dump(record, handle, indent=1)

print("LEFTOVERS {}: {} pydeseq2-* folder(s); joblib temp dir {}; searched {}".format(
	args.label, len(found), record["joblib_temp_dir"], searched))
for item in found:
	print("  {} ({} files: {})".format(item["folder"], len(item["files"]),
		", ".join("{} {} B".format(f["path"], f["bytes"]) for f in item["files"][:10])))
if record["error"]:
	print("  joblib lookup failed: " + record["error"])
