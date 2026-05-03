from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger(__name__)


def shutdown_thread_pool(executor: Any, futures: Iterable[Future[Any]], *, wait: bool) -> None:
    for future in list(futures):
        future.cancel()
    executor.shutdown(wait=wait, cancel_futures=True)


def force_exit_on_keyboard_interrupt(
    task_name: str,
    *,
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    failed_output_path: str | Path | None = None,
) -> None:
    fields: list[str] = [f"{task_name}: 收到 Ctrl + C，立即停止进程。"]
    if output_path is not None:
        fields.append(f"output={Path(output_path)}")
    if checkpoint_path is not None:
        fields.append(f"checkpoint={Path(checkpoint_path)}")
    if failed_output_path is not None:
        fields.append(f"failed_output={Path(failed_output_path)}")
    message = "\r\n".join(fields)
    logger.warning("%s.interrupted force_exit=true", task_name)
    sys.stderr.write(f"{message}\r\n")
    sys.stderr.flush()
    os._exit(130)
