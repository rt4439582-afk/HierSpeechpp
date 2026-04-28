import argparse
from pathlib import Path


def read_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_p = Path(args.base)
    new_p = Path(args.new)
    out_p = Path(args.out)

    lines = []
    seen = set()

    for p in [base_p, new_p]:
        if not p.exists():
            continue
        for line in read_lines(p):
            if line not in seen:
                seen.add(line)
                lines.append(line)

    write_lines(out_p, lines)
    print(f"Wrote {len(lines)} lines -> {out_p}")


if __name__ == "__main__":
    main()
