import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="From pixels to physical-field reconstruction"
    )
    parser.add_argument("--output", default="results/demo")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ml-steps", type=int, default=0)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real photo, with simulated measurements",
    )
    args = parser.parse_args()
    if not 4 <= args.size <= 48 or args.ml_steps < 0:
        parser.error("size must be 4–48 and ml-steps >=0")
    from .benchmark import run

    report = run(args.output, args.size, args.seed, args.ml_steps, args.real)
    print(json.dumps(report["scores"], indent=2))
    print("Visual report:", args.output + "/index.html")


if __name__ == "__main__":
    main()
