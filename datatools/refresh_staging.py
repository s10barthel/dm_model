"""Parent-owned staging paths for independent target-refresh matches."""
from datatools.artifact_storage import staged_path
from project_config import get_resolved_action_dir


def target_refresh_tasks(root, inventory, modes, carries, stack, attempt, *, acquisitions=False):
    for mid, artifacts in inventory.items():
        paths = {artifact.path for artifact in artifacts}
        paths.update(get_resolved_action_dir(mode, root=root, use_carries=carry) / f"{mid}.parquet"
                     for carry in ([False, True] if carries else [False]) for mode in modes)
        if acquisitions:
            paths.add(root / "acquisition_locations" / f"{mid}.parquet")
        temporary = {path: stack.enter_context(staged_path(path, attempt=attempt)) for path in sorted(paths)}
        yield root, mid, artifacts, temporary
