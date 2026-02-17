"""
Quick manual test runner for the diabetes fuzzy engine.
"""

from __future__ import annotations

from modules.diabetes import engine


def _run_case(glucose: float, bmi: float) -> None:
    result = engine.run_inference({"glucose": glucose, "bmi": bmi})
    print(f"glucose={glucose}, bmi={bmi} -> {result}")


def main() -> None:
    # A few representative cases
    _run_case(20, 12)    # expected low-ish risk
    # _run_case(120, 27)   # expected medium-ish risk
    # _run_case(180, 33)   # expected high-ish risk


if __name__ == "__main__":
    main()
