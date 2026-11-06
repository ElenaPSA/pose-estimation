from __future__ import annotations

import logging

from .config import Settings
from .video_processor import process_video


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    settings = Settings()
    process_video(settings)


if __name__ == "__main__":
    run()