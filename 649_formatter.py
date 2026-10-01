#!/usr/bin/env python3
"""
Lotto Draw -> Line Number Finder

Reads the draws CSV, scans the combinations file once, and records the line
number where each draw's six numbers appear. Writes the results as a CSV.

Draws CSV (input):              Combinations file (input):
    number,date                     number
    2 12 23 30 32 35,03-Jan-2024    1 2 3 4 5 6
    18 19 33 38 48 49,06-Jan-2024   ...

Results CSV (output):
    date,line_number,numbers
    03-Jan-24,2794277,2 12 23 30 32 35
    06-Jan-24,13103044,18 19 33 38 48 49

Line numbers count combination lines only (the header line is not counted),
so "1 2 3 4 5 6" is line 1.

Usage:
    python lottery_formatter.py combos.txt
    python lottery_formatter.py C:\\649\\649.txt -c C:\\649\\lottery.csv -o C:\\649\\results.csv
"""

import argparse
import os
import re
import sys
import time
from math import comb

DEFAULT_CSV = r"C:\649\lottery.csv"
N, K = 49, 6

DIGITS = re.compile(rb"\d+")


# ---------------------------------------------------------------- helpers
def make_key(text):
    """b'05 12  2 ...' -> canonical b'2 5 12 ...' (sorted, single spaces), or None."""
    parts = DIGITS.findall(text)
    if len(parts) != K:
        return None
    vals = sorted(int(p) for p in parts)
    if vals[0] < 1 or vals[-1] > N or len(set(vals)) != K:
        return None
    return b" ".join(str(v).encode() for v in vals)


def short_date(date):
    """'03-Jan-2024' -> '03-Jan-24'. Other formats are left unchanged."""
    parts = date.split("-")
    if len(parts) == 3 and len(parts[2]) == 4 and parts[2].isdigit():
        parts[2] = parts[2][2:]
    return "-".join(parts)


def expected_line(key):
    """Position of a combination in standard ascending order (1 2 3 4 5 6 = 1)."""
    vals = [int(v) for v in key.split()]
    r, prev = 1, 0
    for i, x in enumerate(vals):
        for v in range(prev + 1, x):
            r += comb(N - v, K - 1 - i)
        prev = x
    return r


# ---------------------------------------------------------------- loading
def load_draws(path):
    """Return list of draws in CSV order, plus a key -> [draw indexes] lookup."""
    draws, lookup, skipped = [], {}, []
    with open(path, "rb") as f:
        next(f, None)  # header
        for n, raw in enumerate(f, start=2):
            line = raw.strip()
            if not line:
                continue
            numbers, sep, date = line.rpartition(b",")
            key = make_key(numbers) if sep else None
            date = date.strip().decode("utf-8", "replace")
            if key is None or not date:
                skipped.append(n)
                continue
            lookup.setdefault(key, []).append(len(draws))
            draws.append({"date": date, "key": key, "line": None})
    return draws, lookup, skipped


# ---------------------------------------------------------------- scanning
def scan(combo_path, lookup, normalize):
    """Stream the combinations file. Returns (line_numbers dict, data_lines, duplicates)."""
    found = {}          # key -> first line number
    duplicates = {}     # key -> extra line numbers
    get = lookup.get
    data_lines = 0
    first = True

    with open(combo_path, "rb", buffering=16 * 1024 * 1024) as f:
        for raw in f:
            body = raw.rstrip(b"\r\n")

            # Skip a header line such as "number"
            if first:
                first = False
                if make_key(body) is None:
                    continue

            data_lines += 1

            key = body if body in lookup else None          # fast exact match
            if key is None and normalize:
                k = make_key(body)                          # tabs / spaces / order
                if k is not None and k in lookup:
                    key = k

            if key is not None:
                if key in found:
                    duplicates.setdefault(key, []).append(data_lines)
                else:
                    found[key] = data_lines

    return found, data_lines, duplicates


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Find line numbers of lotto draws in a combinations file.")
    ap.add_argument("combos", help="combinations file (header 'number', then one combo per line)")
    ap.add_argument("-c", "--csv", default=DEFAULT_CSV, help="draws CSV (default: %(default)s)")
    ap.add_argument("-o", "--output", help="results CSV (default: results.csv beside the draws CSV)")
    ap.add_argument("--long-year", action="store_true", help="keep 4-digit years (03-Jan-2024)")
    ap.add_argument("--normalize", action="store_true",
                    help="also match lines with tabs, extra spaces or unsorted numbers (slower)")
    args = ap.parse_args()

    for p in (args.combos, args.csv):
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}")

    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(args.csv)), "results.csv")

    draws, lookup, skipped = load_draws(args.csv)
    if not draws:
        sys.exit("No valid draws were loaded from the CSV.")

    start = time.perf_counter()
    found, data_lines, duplicates = scan(args.combos, lookup, args.normalize)
    elapsed = time.perf_counter() - start

    # Write results in the same order as the draws CSV
    written, missing, order_warnings = 0, [], []
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        out.write("date,line_number,numbers\r\n")
        for d in draws:
            line_no = found.get(d["key"])
            if line_no is None:
                missing.append(d)
                continue
            date = d["date"] if args.long_year else short_date(d["date"])
            numbers = d["key"].decode()
            out.write(f"{date},{line_no},{numbers}\r\n")
            written += 1
            if line_no != expected_line(d["key"]):
                order_warnings.append((numbers, line_no, expected_line(d["key"])))

    # ------------------------------------------------------------ summary
    print(f"Draws loaded:        {len(draws)}")
    print(f"Draw rows skipped:   {len(skipped)}" + (f"  (CSV lines {skipped[:10]})" if skipped else ""))
    print(f"Combination lines:   {data_lines:,}")
    print(f"Draws found:         {written}")
    print(f"Draws not found:     {len(missing)}")
    print(f"Scan time:           {elapsed:.1f} s")
    print(f"Results:             {out_path}")

    if missing:
        print("\nNot found" + (" (first 20)" if len(missing) > 20 else "") + ":")
        for d in missing[:20]:
            print(f"  {d['key'].decode()}  ({d['date']})")
        if not args.normalize:
            print("  Tip: try --normalize if lines use tabs, extra spaces or unsorted numbers.")

    if duplicates:
        print("\nCombinations appearing more than once (first line number used):")
        for k, extra in list(duplicates.items())[:20]:
            print(f"  {k.decode()}  first at {found[k]}, also at {extra[:5]}")

    repeats = {k: v for k, v in lookup.items() if len(v) > 1}
    if repeats:
        print("\nSame numbers drawn on more than one date (each gets its own row):")
        for k, idxs in repeats.items():
            print(f"  {k.decode()}  " + ", ".join(draws[i]["date"] for i in idxs))

    if order_warnings and data_lines == comb(N, K):
        print("\nNote: some line numbers differ from standard ascending combination order:")
        for nums, got, exp in order_warnings[:10]:
            print(f"  {nums}  found {got}, expected {exp}")


if __name__ == "__main__":
    main()