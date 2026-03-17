from __future__ import annotations

import json

from src.modeling import MODEL_PATH, train_and_persist


def main() -> None:
    bundle = train_and_persist()
    summary = {
        "model_name": bundle["model_name"],
        "model_path": str(MODEL_PATH),
        "metrics": bundle["metrics"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
