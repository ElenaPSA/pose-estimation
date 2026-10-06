from __future__ import annotations

import logging

from .config import Settings
from .video_processor import process_video


logger = logging.getLogger(__name__)


def main() -> None:
    settings = Settings()
    settings.validate()

    logger.info(
        "Starting local driver pose estimation"
    )

    logger.info(
        "Camera index=%d model=%s device=%s",
        settings.camera_index,
        settings.model_path,
        settings.device,
    )

    process_video(settings)


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(levelname)s - "
            "%(name)s - %(message)s"
        ),
    )

    try:
        main()

    except KeyboardInterrupt:
        logger.info(
            "Application terminated by user"
        )

    except Exception:
        logger.exception(
            "Application failed"
        )

        raise


if __name__ == "__main__":
    run()