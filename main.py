import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_DIR = Path(__file__).parent / "data"
PROJECT_ROOT = Path(__file__).parent

MENU_OPTIONS = [
    ("1", "Generate dataset"),
    ("2", "Analyze dialogs"),
    ("3", "Verify with second model (optional)"),
    ("4", "Evaluate metrics"),
    ("5", "Run full pipeline (generate -> analyze -> evaluate)"),
    ("6", "Run tests"),
    ("0", "Exit"),
]


def timed(label):
    class Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"\n{'=' * 65}")
            print(f"  {label}")
            print(f"{'=' * 65}")
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            mins, secs = divmod(elapsed, 60)
            if mins > 0:
                print(f"\n  [{label}] completed in {int(mins)}m {secs:.1f}s")
            else:
                print(f"\n  [{label}] completed in {secs:.1f}s")
            self.elapsed = elapsed
    return Timer()


def cmd_generate():
    with timed("Generate dataset"):
        from src.generate import main
        main()


def cmd_analyze(voting_rounds=3):
    with timed(f"Analyze dialogs (voting x{voting_rounds})"):
        from src.analyze import main
        main(voting_rounds=voting_rounds)


def cmd_verify():
    with timed("Verify with second model"):
        from src.verify import run_verification
        run_verification()


def cmd_evaluate(file_path=None):
    from src.evaluate import evaluate

    if file_path:
        with timed("Evaluate"):
            evaluate(file_path)
        return

    files = [
        ("analysis.json",          "Voting only"),
        ("analysis_verified.json", "After verify"),
        ("analysis_hybrid.json",   "Hybrid"),
    ]
    available = [(f, label) for f, label in files if (DATA_DIR / f).exists()]

    if not available:
        print("No analysis files found. Run analyze first.")
        return

    if len(available) == 1:
        with timed("Evaluate"):
            evaluate(str(DATA_DIR / available[0][0]))
        return

    print("\n  Available analysis files:")
    for i, (f, label) in enumerate(available, 1):
        print(f"    [{i}] {label:<20} ({f})")
    print(f"    [a] Evaluate all")

    try:
        choice = input("\n  Select file: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "a":
        for f, label in available:
            with timed(f"Evaluate: {label}"):
                evaluate(str(DATA_DIR / f))
    elif choice.isdigit() and 1 <= int(choice) <= len(available):
        f, label = available[int(choice) - 1]
        with timed(f"Evaluate: {label}"):
            evaluate(str(DATA_DIR / f))
    else:
        print("Invalid choice.")


def cmd_run(voting_rounds=3):
    total_start = time.time()

    cmd_generate()
    cmd_analyze(voting_rounds=voting_rounds)
    cmd_evaluate(file_path=str(DATA_DIR / "analysis.json"))

    total = time.time() - total_start
    mins, secs = divmod(total, 60)
    print(f"\n{'=' * 65}")
    print(f"  PIPELINE COMPLETE  |  Total time: {int(mins)}m {secs:.1f}s")
    print(f"{'=' * 65}")


def cmd_test():
    with timed("Run tests"):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v"],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"\n  Tests failed with exit code {result.returncode}")


def show_menu():
    print(f"\n{'=' * 65}")
    print("  Support Chat Quality Analysis Pipeline")
    print(f"{'=' * 65}")
    for key, label in MENU_OPTIONS:
        print(f"  [{key}] {label}")
    print(f"{'-' * 65}")


def interactive():
    while True:
        show_menu()

        try:
            choice = input("\n  Select option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "0":
            print("Exiting.")
            break

        commands = {
            "1": cmd_generate,
            "2": cmd_analyze,
            "3": cmd_verify,
            "4": cmd_evaluate,
            "5": cmd_run,
            "6": cmd_test,
        }

        if choice in commands:
            try:
                commands[choice]()
            except KeyboardInterrupt:
                print("\n\n  Cancelled.")
        else:
            print("  Invalid option.")


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        rounds = 3
        for i, arg in enumerate(sys.argv):
            if arg == "--rounds" and i + 1 < len(sys.argv):
                rounds = int(sys.argv[i + 1])

        dispatch = {
            "generate":  lambda: cmd_generate(),
            "analyze":   lambda: cmd_analyze(voting_rounds=rounds),
            "verify":    lambda: cmd_verify(),
            "evaluate":  lambda: cmd_evaluate(),
            "run":       lambda: cmd_run(voting_rounds=rounds),
            "test":      lambda: cmd_test(),
        }

        if cmd in dispatch:
            dispatch[cmd]()
        elif cmd in ("-h", "--help"):
            print("Usage: python main.py [command] [--rounds N]")
            print("\nCommands:")
            print("  generate            Generate chat dataset")
            print("  analyze             Analyze dialogs with LLM")
            print("  verify              Verify with second model (optional)")
            print("  evaluate            Compute accuracy metrics")
            print("  run                 Full pipeline: generate -> analyze -> evaluate")
            print("  test                Run unit tests")
            print("\nOptions:")
            print("  --rounds N          Voting rounds for analyze (1=fast, 3=accurate, default: 3)")
            print("\nRun without arguments for interactive mode.")
        else:
            print(f"Unknown command: {cmd}")
            print("Run with --help for available commands.")
    else:
        interactive()


if __name__ == "__main__":
    main()
