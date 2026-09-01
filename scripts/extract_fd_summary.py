#!/usr/bin/env python3
import argparse, csv, glob, os, re

def find_fd_files(input_dir, pattern, exclude):
    files = glob.glob(os.path.join(input_dir, "**", pattern), recursive=True)
    if exclude:
        files = [f for f in files if exclude not in os.path.basename(f)]
    return sorted(files)

def extract_subject_id(filename):
    base = os.path.basename(filename)
    m = re.search(r"(sub-[A-Za-z0-9]+)", base)
    return m.group(1) if m else base

def summarize_fd_file(filepath, skip_t0=True):
    meanfd, maxfd = [], []
    with open(filepath, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if skip_t0 and i == 0:
                continue
            meanfd.append(float(row["MeanFD"]))
            maxfd.append(float(row["MaxFD"]))
    if not meanfd:
        return None
    n = len(meanfd)
    return {"n_timepoints": n, "MeanFD_avg": sum(meanfd)/n, "MaxFD_avg": sum(maxfd)/n}

def main():
    ap = argparse.ArgumentParser(description="Summarize RABIES FD csv files across subjects.")
    ap.add_argument("input_dir")
    ap.add_argument("-o", "--output", default="fd_summary.csv")
    ap.add_argument("--pattern", default="*_FD_file.csv")
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--keep-t0", action="store_true")
    args = ap.parse_args()
    files = find_fd_files(args.input_dir, args.pattern, args.exclude)
    if not files:
        print(f"No files matching '{args.pattern}' found in {args.input_dir}")
        return
    rows = []
    for f in files:
        stats = summarize_fd_file(f, skip_t0=not args.keep_t0)
        if stats is None:
            print(f"WARNING: {f} produced no rows, skipping")
            continue
        stats["subject"] = extract_subject_id(f)
        stats["file"] = os.path.basename(f)
        rows.append(stats)
    fieldnames = ["subject", "file", "n_timepoints", "MeanFD_avg", "MaxFD_avg"]
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} subjects to {args.output}")

if __name__ == "__main__":
    main()
