"""Compare main's and the optimization's outputs (fork-only CI helper).

  compare.py make-input OUT.npz
	  The synthetic input: negative binomial counts, 40 samples x 8,000 genes,
	  conditions A and B (20 each), numpy.random.default_rng(0).
  compare.py run --data example|synthetic [--input OUT.npz] --n-cpus N --out DIR
	  With the pydeseq2 installed in the running Python: DeseqDataSet(...).deseq2(),
	  DeseqStats(contrast=["condition", "B", "A"]).summary() and
	  lfc_shrink(coeff="condition[T.B]"); saves every output a user can read
	  (obs, var, varm, obsm, layers and uns of the DeseqDataSet, results_df before
	  and after lfc_shrink) to DIR/outputs.npz, warnings and errors to
	  DIR/record.json.
  compare.py compare --ref DIR --new DIR --label TEXT --tsv FILE
	  Per output: bitwise identical, or the largest relative difference
	  |new - ref| / |ref| (0 where equal; a NaN on one side only, or a shape
	  change, is inf). Appends rows to FILE and prints a summary.
"""

import argparse
import json
import os
import sys
import time
import warnings


def make_input(out):
	import numpy

	rng = numpy.random.default_rng(0)
	n_samples, n_genes = 40, 8000
	base = numpy.exp(rng.normal(4.0, 1.8, n_genes))
	dispersion = (0.04 + 2.0 / base) * numpy.exp(rng.normal(0.0, 0.5, n_genes))
	lfc = numpy.where(rng.random(n_genes) < 0.1, rng.normal(0.0, 1.5, n_genes), 0.0)
	condition = numpy.repeat([0, 1], n_samples // 2)
	size_factors = numpy.exp(rng.normal(0.0, 0.25, n_samples))
	mu = size_factors[:, None] * base[None, :] * numpy.exp2(lfc[None, :] * condition[:, None])
	r = numpy.broadcast_to(1.0 / dispersion, mu.shape)
	counts = rng.negative_binomial(r, r / (r + mu)).astype(numpy.int64)
	numpy.savez(out, counts=counts, condition=numpy.array(["A", "B"])[condition])
	print("synthetic input {}: counts {} {}, total {}, zero genes {}".format(
		out, counts.shape, counts.dtype, int(counts.sum()), int((counts.sum(0) == 0).sum())))


def cpu_name():
	"""The runner's CPU model, as the OS reports it."""
	import platform
	import subprocess

	try:
		if sys.platform == "darwin":
			return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.model"],
				capture_output=True, text=True).stdout.strip().replace("\n", " / ")
		if sys.platform.startswith("linux"):
			with open("/proc/cpuinfo") as handle:
				for line in handle:
					if line.startswith("model name"):
						return line.split(":", 1)[1].strip()
		return "{} / {}".format(os.environ.get("PROCESSOR_IDENTIFIER"), platform.processor())
	except Exception as exc:  # noqa: BLE001
		return repr(exc)


def load(data, path):
	import numpy
	import pandas

	if data == "example":
		from pydeseq2.datasets import load_example_data

		return load_example_data(modality="raw_counts"), load_example_data(modality="metadata")
	with numpy.load(path) as handle:
		counts, condition = handle["counts"], handle["condition"]
	samples = ["sample{}".format(i + 1) for i in range(counts.shape[0])]
	genes = ["gene{}".format(j + 1) for j in range(counts.shape[1])]
	return (pandas.DataFrame(counts, index=samples, columns=genes),
		pandas.DataFrame({"condition": condition.astype(str)}, index=samples))


def as_array(value):
	import numpy
	import pandas

	if isinstance(value, (pandas.DataFrame, pandas.Series)):
		value = value.to_numpy()
	a = numpy.asarray(value)
	if a.dtype == object:
		try:
			return a.astype(numpy.float64)
		except (TypeError, ValueError):
			return numpy.asarray([str(x) for x in a.ravel()], dtype=str).reshape(a.shape)
	if a.dtype.kind in "biuf":
		return a.astype(numpy.float64)
	return numpy.asarray([str(x) for x in a.ravel()], dtype=str).reshape(a.shape)


def collect(dds, unshrunk, shrunk):
	import numpy

	out = {}
	for name, frame in [("obs", dds.obs), ("var", dds.var)]:
		for col in frame.columns:
			out["{}/{}".format(name, col)] = as_array(frame[col])
		out["{}/_index".format(name)] = numpy.asarray([str(i) for i in frame.index], dtype=str)
	for name, mapping in [("varm", dds.varm), ("obsm", dds.obsm), ("layers", dds.layers)]:
		for key in mapping.keys():
			out["{}/{}".format(name, key)] = as_array(mapping[key])
	for key, value in dds.uns.items():
		items = value.items() if isinstance(value, dict) else [(None, value)]
		for k, v in items:
			name = "uns/{}".format(key) if k is None else "uns/{}/{}".format(key, k)
			try:
				out[name] = as_array(v) if not isinstance(v, (dict, list)) else numpy.asarray(repr(v))
			except Exception:  # noqa: BLE001
				out[name] = numpy.asarray(repr(v))
	for prefix, frame in [("results", unshrunk), ("shrunk", shrunk)]:
		for col in frame.columns:
			out["{}/{}".format(prefix, col)] = as_array(frame[col])
	return out


def run(args):
	import numpy
	import pydeseq2
	from pydeseq2.dds import DeseqDataSet
	from pydeseq2.ds import DeseqStats

	os.makedirs(args.out, exist_ok=True)
	counts_df, metadata = load(args.data, args.input)
	record = {"pydeseq2_file": pydeseq2.__file__, "python": sys.version, "numpy": numpy.__version__,
		"data": args.data, "n_cpus": args.n_cpus, "shape": list(counts_df.shape),
		"thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
			"VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}}
	try:
		import threadpoolctl

		record["threadpools"] = threadpoolctl.threadpool_info()
	except Exception as exc:  # noqa: BLE001
		record["threadpools"] = repr(exc)
	record["cpu"] = cpu_name()
	outputs, error = {}, None
	t0 = time.perf_counter()
	with warnings.catch_warnings(record=True) as caught:
		warnings.simplefilter("always")
		try:
			dds = DeseqDataSet(counts=counts_df, metadata=metadata, design="~condition",
				n_cpus=args.n_cpus, quiet=True)
			dds.deseq2()
			ds = DeseqStats(dds, contrast=["condition", "B", "A"], n_cpus=args.n_cpus, quiet=True)
			ds.summary()
			unshrunk = ds.results_df.copy()
			ds.lfc_shrink(coeff="condition[T.B]")
			record["seconds"] = time.perf_counter() - t0
			outputs = collect(dds, unshrunk, ds.results_df)
		except Exception as exc:  # noqa: BLE001
			error = {"type": type(exc).__name__, "message": str(exc)}
	record["error"] = error
	record["warnings"] = sorted({"{}: {}".format(w.category.__name__, w.message) for w in caught})
	record["keys"] = sorted(outputs)
	numpy.savez_compressed(os.path.join(args.out, "outputs.npz"),
		**{k.replace("/", "|"): v for k, v in outputs.items()})
	with open(os.path.join(args.out, "record.json"), "w") as handle:
		json.dump(record, handle, indent=1)
	print("{} n_cpus={} {}: {} outputs, {:.1f} s, error {}, {} warning(s); cpu {}".format(args.data, args.n_cpus,
		pydeseq2.__file__, len(outputs), record.get("seconds") or float("nan"), error, len(record["warnings"]),
		record["cpu"]))


def relative(a, b):
	"""Largest |a - b| / |b|, NaN-aware; inf for a one-sided NaN or b == 0 != a."""
	import numpy

	a = numpy.asarray(a, dtype=float).ravel()
	b = numpy.asarray(b, dtype=float).ravel()
	na, nb = numpy.isnan(a), numpy.isnan(b)
	with numpy.errstate(invalid="ignore", divide="ignore"):
		diff = numpy.where(a == b, 0.0, numpy.abs(a - b))
		rel = numpy.where(diff == 0, 0.0, diff / numpy.abs(b))
	rel = numpy.where(na & nb, 0.0, numpy.where(na | nb, numpy.inf, rel))
	return float(rel.max()) if rel.size else 0.0, int((rel > 0).sum())


def compare(args):
	import numpy

	def read(folder):
		with open(os.path.join(folder, "record.json")) as handle:
			record = json.load(handle)
		with numpy.load(os.path.join(folder, "outputs.npz"), allow_pickle=False) as data:
			arrays = {k.replace("|", "/"): data[k] for k in data.files}
		return record, arrays

	rm, am = read(args.ref)
	ro, ao = read(args.new)
	rows = []
	for key in sorted(set(am) | set(ao)):
		if key not in ao or key not in am:
			rows.append((key, "missing in " + ("new" if key not in ao else "ref"), "", "", ""))
			continue
		a, b = ao[key], am[key]
		if a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes():
			rows.append((key, "bitwise", "0", "0", str(b.size)))
		elif a.shape != b.shape:
			rows.append((key, "shape {} vs {}".format(a.shape, b.shape), "inf", "", str(b.size)))
		elif a.dtype.kind in "US" or b.dtype.kind in "US":
			n = int((a.astype(str) != b.astype(str)).sum())
			rows.append((key, "text differs", "inf", str(n), str(b.size)))
		else:
			worst, n = relative(a, b)
			rows.append((key, "differs", repr(worst), str(n), str(b.size)))
	rows.append(("error", "same" if rm["error"] == ro["error"] else "differs: ref {} new {}".format(
		rm["error"], ro["error"]), "", "", ""))
	rows.append(("warnings", "same" if rm["warnings"] == ro["warnings"] else "differs: ref {} new {}".format(
		rm["warnings"], ro["warnings"]), "", "", ""))
	rows.append(("seconds", "ref {:.2f} new {:.2f}".format(rm.get("seconds") or float("nan"),
		ro.get("seconds") or float("nan")), "", "", ""))
	new = not os.path.exists(args.tsv)
	with open(args.tsv, "a") as handle:
		if new:
			handle.write("label\toutput\tstatus\tmax_rel\tn_differ\tn_elements\n")
		for row in rows:
			handle.write("\t".join((args.label,) + row) + "\n")
	arrays = [r for r in rows if r[0] not in ("error", "warnings", "seconds")]
	n_bitwise = sum(r[1] == "bitwise" for r in arrays)
	worst = max([float(r[2]) for r in arrays if r[2] not in ("", "0")] or [0.0])
	print("COMPARE {}: {}/{} outputs bitwise identical; largest relative difference {:.3g}; error {}; warnings {}".format(
		args.label, n_bitwise, len(arrays), worst, rows[-3][1], rows[-2][1]))
	for r in arrays:
		if r[1] != "bitwise":
			print("  {}: {} max_rel {} ({} of {} elements)".format(*r))


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
p = sub.add_parser("make-input")
p.add_argument("out")
p = sub.add_parser("run")
p.add_argument("--data", choices=["example", "synthetic"], required=True)
p.add_argument("--input")
p.add_argument("--n-cpus", type=int, required=True)
p.add_argument("--out", required=True)
p = sub.add_parser("compare")
p.add_argument("--ref", required=True)
p.add_argument("--new", required=True)
p.add_argument("--label", required=True)
p.add_argument("--tsv", required=True)
args = parser.parse_args()
if args.command == "make-input":
	make_input(args.out)
elif args.command == "run":
	run(args)
else:
	compare(args)
