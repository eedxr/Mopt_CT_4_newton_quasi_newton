from __future__ import annotations

from experiments import start_experiments, summarize_to_console


def main() -> None:
    summary = start_experiments()
    summarize_to_console(summary)


if __name__ == "__main__":
    main()
