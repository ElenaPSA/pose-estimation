from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            value,
            output_file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    temporary_path.replace(path)