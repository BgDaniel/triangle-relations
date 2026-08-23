"""Shared joblib/tqdm plumbing used by both discovery pipelines (Program 1 and 1b)."""

from __future__ import annotations

import contextlib
from typing import Iterator

import joblib
from tqdm.auto import tqdm


@contextlib.contextmanager
def joblib_progress(total: int, desc: str) -> Iterator[tqdm]:
    """Report :class:`joblib.Parallel` batch completions into a tqdm bar.

    ``joblib.Parallel`` has no built-in progress callback; this patches its
    batch-completion hook for the duration of the ``with`` block so each
    finished item (including ones run in other worker processes) ticks the
    bar, then restores the original hook.
    """
    progress_bar = tqdm(total=total, desc=desc, unit="triple")

    class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress_bar.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        progress_bar.close()
