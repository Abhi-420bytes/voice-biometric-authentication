"""
Verifies all required packages and prints version info.
Run this to confirm Phase 1 environment is ready.
"""
import sys

checks = []

def check(name, fn):
    try:
        result = fn()
        checks.append((name, True, result))
    except Exception as e:
        checks.append((name, False, str(e)))

check("Python",         lambda: sys.version.split()[0])
check("numpy",          lambda: __import__('numpy').__version__)
check("scipy",          lambda: __import__('scipy').__version__)
check("torch",          lambda: __import__('torch').__version__)
check("librosa",        lambda: __import__('librosa').__version__)
check("sounddevice",    lambda: __import__('sounddevice').__version__)
check("soundfile",      lambda: __import__('soundfile').__version__)
check("noisereduce",    lambda: getattr(__import__('noisereduce'), '__version__', 'installed'))
check("fastapi",        lambda: __import__('fastapi').__version__)
check("sklearn",        lambda: __import__('sklearn').__version__)
check("matplotlib",     lambda: __import__('matplotlib').__version__)
check("sqlite3",        lambda: __import__('sqlite3').sqlite_version)

print("\n=== Environment Verification ===")
all_ok = True
for name, ok, info in checks:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name:<15} {info}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("All checks passed. Phase 1 environment is ready.")
else:
    print("Some packages are missing. Run: pip install -r requirements.txt")
