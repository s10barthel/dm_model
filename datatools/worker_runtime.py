"""Bounded process workers for independent feature matches and files."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager, closing
import multiprocessing
import os


def resolve_num_workers(value):
    if isinstance(value, int):
        workers = value
    else:
        text = str(value).strip().lower()
        workers = max(1, min(6, (os.cpu_count() or 1) - 2)) if text == "auto" else int(text)
    if workers < 1:
        raise ValueError("--num-workers must be a positive integer or 'auto'.")
    return workers


def configure_worker_thread_limit(limit):
    if int(limit) < 1:
        raise ValueError("--worker-thread-limit must be a positive integer.")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(int(limit))


@contextmanager
def _worker_environment(limit):
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    configure_worker_thread_limit(limit)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_worker(function, shared_args, unpack, thread_limit):
    global _function, _shared_args, _unpack
    import torch
    from threadpoolctl import threadpool_limits
    configure_worker_thread_limit(thread_limit)
    torch.set_num_threads(thread_limit)
    threadpool_limits(limits=thread_limit)
    _function, _shared_args, _unpack = function, shared_args, unpack


def _run_task(task):
    args = task if _unpack else (task,)
    return _function(*args, *_shared_args)


def _call_shared_task(task, function, fields, unpack):
    values = list(task)
    for index, value in fields:
        values.insert(index, value)
    return function(*values) if unpack else function(tuple(values))


def map_shared_tasks(function, tasks, num_workers=1, worker_thread_limit=1, *, shared_indices=(2,), unpack=False):
    """Keep large, immutable tuple fields in the worker instead of every submission."""
    resolve_num_workers(num_workers)
    if int(worker_thread_limit) < 1:
        raise ValueError("--worker-thread-limit must be a positive integer.")
    tasks = iter(tasks)
    first = next(tasks, None)
    if first is None:
        return
    indices = tuple(sorted(shared_indices))
    fields = tuple((index, first[index]) for index in indices)

    def compact():
        yield tuple(value for index, value in enumerate(first) if index not in indices)
        for task in tasks:
            if any(task[index] is not value and task[index] != value for index, value in fields):
                raise ValueError("Shared worker configuration differs between tasks.")
            yield tuple(value for index, value in enumerate(task) if index not in indices)

    with closing(map_tasks(_call_shared_task, compact(), num_workers,
            worker_thread_limit, shared_args=(function, fields, unpack))) as results:
        yield from results


def map_tasks(function, tasks, num_workers=1, worker_thread_limit=1, *, shared_args=(), unpack=False):
    """Yield in input order; schedule by completion and send shared inputs only once.

    The caller must close this iterator before cleaning up workers' staged files.
    Running tasks finish before exceptions escape, so cleanup cannot race writes.
    """
    workers = resolve_num_workers(num_workers)
    if int(worker_thread_limit) < 1:
        raise ValueError("--worker-thread-limit must be a positive integer.")
    tasks = iter(tasks)
    # Avoid importing native runtimes or starting processes for empty workloads.
    initial = []
    for _ in range(workers):
        task = next(tasks, None)
        if task is None:
            break
        initial.append(task)
    if not initial:
        return
    workers = min(workers, len(initial))
    if workers == 1:
        for task in initial:
            args = task if unpack else (task,)
            yield function(*args, *shared_args)
        for task in tasks:
            args = task if unpack else (task,)
            yield function(*args, *shared_args)
        return
    with _worker_environment(worker_thread_limit):
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(function, shared_args, unpack, worker_thread_limit)) as pool:
            pending, ready = {}, {}
            submitted, yielded = 0, 0
            try:
                for task in initial:
                    pending[pool.submit(_run_task, task)] = submitted
                    submitted += 1
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        try:
                            ready[index] = future.result()
                        except BaseException as error:
                            error.feature_task_index = index
                            error.add_note(f"Feature worker task index: {index}")
                            raise
                        task = next(tasks, None)
                        if task is not None:
                            pending[pool.submit(_run_task, task)] = submitted
                            submitted += 1
                    while yielded in ready:
                        yield ready.pop(yielded)
                        yielded += 1
            finally:
                for future in pending:
                    future.cancel()
