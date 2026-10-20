"""Single entry point for the platform: `python solution.py [task.md]`.

Reads the task files, runs the autonomous research agent and writes submission.csv.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import Config  # noqa: E402
from agent.pipeline import run  # noqa: E402


def main():
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(Config.from_env(" ".join(sys.argv[1:]) or None   # tolerate unquoted paths with spaces))


if __name__ == "__main__":
    main()
