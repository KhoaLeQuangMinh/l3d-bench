"""Google Drive sync through rclone, ported from your UNETR++ notebook (unetr.ipynb, cell 0).

Same behaviour: remote `gdrive`, single-line configs are re-split into lines, every call retries with
exponential backoff, uploads use 128M chunks. Disabled automatically (local-only) when no config is found.

The config (it holds a Google refresh token) must never be committed: this repo is public. On Kaggle it is
read from the account's secret RCLONE_CONFIG_GDRIVE (Add-ons > Secrets), the same as the UNETR++ notebook.

Remote layout: gdrive:l3d-bench/results/<run_id>/...
"""
from __future__ import annotations

import itertools
import os
import re
import subprocess
import threading
import time
from pathlib import Path

CONFIG_PATHS = ("/root/.config/rclone/rclone.conf", "/root/.rclone.conf")


def format_rclone_config(raw: str, remote: str = "gdrive") -> str:
    """Turn a pasted secret (possibly flattened onto one line) into a valid rclone.conf section."""
    text = raw.strip()
    if not text.startswith("["):
        text = f"[{remote}]\n" + text
    if "\n" not in text:
        text = text.replace(f"[{remote}]", f"[{remote}]\n")
    for key in ("type", "scope", "token", "client_id", "client_secret", "root_folder_id", "team_drive"):
        text = re.sub(rf"[ \t]+({key}\s*=)", r"\n\1", text)
    return text if text.endswith("\n") else text + "\n"


class DriveSync:
    def __init__(self, remote: str = "gdrive", base_dir: str = "l3d-bench/results",
                 enabled: bool | None = None, runner=subprocess.run, sleep=time.sleep):
        self.remote, self.base_dir = remote, base_dir.strip("/")
        self._run, self._sleep = runner, sleep
        self.enabled = self._detect() if enabled is None else enabled
        # background uploads: one worker, in order, newest copy of each file wins
        self._pending: dict[tuple[str, str], Path] = {}
        self._order: list[tuple[str, str]] = []
        self._cv = threading.Condition()
        self._worker: threading.Thread | None = None
        self._busy = False
        self._counter = itertools.count()
        self.failed: list[str] = []

    # ---- setup ---------------------------------------------------------------------------------
    @staticmethod
    def _detect() -> bool:
        cands = list(CONFIG_PATHS) + [os.path.expanduser("~/.config/rclone/rclone.conf")]
        return any(Path(p).exists() for p in cands)

    @staticmethod
    def _install(raw: str, remote: str, targets=CONFIG_PATHS) -> None:
        text = format_rclone_config(raw, remote)
        for p in targets:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_text(text, encoding="utf-8")
            os.chmod(p, 0o600)

    @classmethod
    def from_kaggle(cls, secret_name: str = "RCLONE_CONFIG_GDRIVE", targets=CONFIG_PATHS,
                    get_secret=None, **kw) -> DriveSync:
        """Install the rclone config from the Kaggle secret, so it never lives in the (public) repo.
        Falls back to a config already on the machine; otherwise sync is disabled and results stay local."""
        remote = kw.get("remote", "gdrive")
        try:
            if get_secret is None:
                from kaggle_secrets import UserSecretsClient  # only exists on Kaggle
                get_secret = UserSecretsClient().get_secret
            raw = get_secret(secret_name)
            if not raw or not raw.strip():
                raise ValueError("secret is empty")
            cls._install(raw, remote, targets)
            return cls(enabled=True, **kw)
        except Exception as exc:  # noqa: BLE001 - any failure means "no secret here"
            sync = cls(**kw)
            if not sync.enabled:
                print(f"[drive] secret {secret_name} not available ({type(exc).__name__}): "
                      "turn it on under Add-ons > Secrets. Results stay local.")
            return sync

    from_kaggle_secret = from_kaggle  # old name

    # ---- primitives ----------------------------------------------------------------------------
    def remote_path(self, run_id: str, name: str = "") -> str:
        tail = f"{self.base_dir}/{run_id}" + (f"/{name}" if name else "")
        return f"{self.remote}:{tail}"

    def _exec(self, args: list[str], attempts: int = 8, backoff: float = 10.0) -> bool:
        """Retries with backoff 10, 20, 40, 80, 120, 120, 120 s (~8.5 min): enough to ride out Drive's per-minute
        'Quota exceeded for quota metric Queries' errors when several lab runs upload at once. --tpslimit keeps
        each process under 4 API calls per second."""
        if not self.enabled:
            return False
        args = args + ["--tpslimit", "4"]
        for attempt in range(1, attempts + 1):
            res = self._run(["rclone"] + args, capture_output=True, text=True)
            if res.returncode == 0:
                return True
            print(f"[drive] attempt {attempt}/{attempts} failed: {res.stderr.strip()[:120]}")
            if attempt < attempts:
                self._sleep(backoff)
                backoff = min(backoff * 2.0, 120.0)
        return False

    def push_file(self, local: str | Path, run_id: str) -> bool:
        if not Path(local).exists():
            return False
        return self._exec(["copy", str(local), self.remote_path(run_id),
                           "--drive-chunk-size", "128M", "--drive-upload-cutoff", "128M"])

    # ---- background uploads ------------------------------------------------------------------
    def push_file_async(self, local: str | Path, run_id: str) -> bool:
        """Queue an upload and return at once. A hard link freezes the file's current content, so the caller
        may overwrite the file right away. If an older copy of the same file is still waiting, only the
        newest is uploaded. Files go up in the order of their latest request, so a checkpoint never reaches
        Drive before the best.pth it refers to."""
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        stage_dir = local.parent / ".upload"
        stage_dir.mkdir(exist_ok=True)
        snap = stage_dir / f"{local.name}.{next(self._counter)}"
        try:
            os.link(local, snap)
        except OSError:
            import shutil
            shutil.copy2(local, snap)
        key = (run_id, local.name)
        with self._cv:
            old = self._pending.get(key)
            self._pending[key] = snap
            if old is not None:
                self._order.remove(key)
            self._order.append(key)            # order of the latest request: best.pth stays before checkpoint
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._drain, daemon=True)
                self._worker.start()
            self._cv.notify_all()
        if old is not None:
            old.unlink(missing_ok=True)
        return True

    def _drain(self) -> None:
        while True:
            with self._cv:
                if not self._order:
                    self._busy = False
                    self._cv.notify_all()
                    return
                key = self._order.pop(0)
                snap = self._pending.pop(key)
                self._busy = True
            run_id, name = key
            ok = self._exec(["copyto", str(snap), self.remote_path(run_id, name),
                             "--drive-chunk-size", "128M", "--drive-upload-cutoff", "128M"])
            if not ok:
                self.failed.append(name)
            snap.unlink(missing_ok=True)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every queued upload is finished. Returns False on timeout."""
        end = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._order or self._busy:
                left = None if end is None else end - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def push_dir(self, local_dir: str | Path, run_id: str) -> bool:
        return self._exec(["copy", str(local_dir), self.remote_path(run_id), "--drive-chunk-size", "128M"])

    def pull_file(self, run_id: str, name: str, local_dir: str | Path) -> bool:
        return self._exec(["copy", self.remote_path(run_id, name), str(local_dir), "--drive-chunk-size", "128M"])

    def exists(self, run_id: str, name: str) -> bool:
        if not self.enabled:
            return False
        res = self._run(["rclone", "lsf", self.remote_path(run_id), "--include", name],
                        capture_output=True, text=True)
        return res.returncode == 0 and name in res.stdout
