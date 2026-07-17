#!/usr/bin/env python3
"""
Run all integration tests and print a formatted summary.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py --verbose
"""
import subprocess, sys, os, argparse, time

TESTS_DIR = os.path.dirname(__file__)
ROOT      = os.path.join(TESTS_DIR, "..")

SUITES = [
    ("Core Modules",    "tests/test_core.py"),
    ("API Endpoints",   "tests/test_api.py"),
    ("E2E Pipeline",    "tests/test_pipeline.py"),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def run_suite(name, path, verbose):
    args = [sys.executable, "-m", "pytest", path, "-v" if verbose else "-q",
            "--tb=short", "--no-header", "--color=yes"]
    t0  = time.time()
    res = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0

    lines   = (res.stdout + res.stderr).splitlines()
    passed  = sum(1 for l in lines if " PASSED" in l or "passed" in l.lower())
    failed  = sum(1 for l in lines if " FAILED" in l or "failed" in l.lower() or "ERROR" in l)

    # extract pytest summary line
    summary = next((l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l), "")

    return {
        "name":    name,
        "path":    path,
        "passed":  res.returncode == 0,
        "summary": summary.strip(),
        "elapsed": elapsed,
        "output":  res.stdout + res.stderr,
        "rc":      res.returncode,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print(f"\n{BOLD}{CYAN}{'='*58}{RESET}")
    print(f"{BOLD}{CYAN}  Voice Biometric Auth — Integration Test Suite{RESET}")
    print(f"{BOLD}{CYAN}{'='*58}{RESET}\n")

    results = []
    for name, path in SUITES:
        print(f"  Running {CYAN}{name}{RESET} ...", end=" ", flush=True)
        r = run_suite(name, path, args.verbose)
        results.append(r)
        status = f"{GREEN}PASS{RESET}" if r["passed"] else f"{RED}FAIL{RESET}"
        print(f"{status}  ({r['elapsed']:.1f}s)")
        if args.verbose or not r["passed"]:
            for line in r["output"].splitlines():
                print(f"    {line}")

    # Summary table
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed

    print(f"\n{BOLD}{'─'*58}{RESET}")
    print(f"  {'Suite':<25} {'Result':>8}  {'Time':>6}  Details")
    print(f"{'─'*58}")
    for r in results:
        col    = GREEN if r["passed"] else RED
        status = "PASS" if r["passed"] else "FAIL"
        summ   = r["summary"][:30] if r["summary"] else ""
        print(f"  {r['name']:<25} {col}{status}{RESET}  {r['elapsed']:>5.1f}s  {summ}")

    print(f"{'─'*58}")
    overall = f"{GREEN}ALL PASSED{RESET}" if failed == 0 else f"{RED}{failed} FAILED{RESET}"
    print(f"  Total: {passed}/{total} suites passed  —  {overall}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
