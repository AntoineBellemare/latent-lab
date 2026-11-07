"""What the lab holds in memory, and letting it go when memory runs short.

Every model registers when it loads (`Models.loading`), with how to let it go, and is marked each
time it is used (`used`). Before a load, and every few seconds, the manager makes room: while the
system's free commit (RAM plus page file: what Windows refuses an allocation against, and the cause
of "Unable to allocate" and of crashes inside a load) or the card's free memory is below its margin,
the least recently used model that has not been touched for `PROTECT` seconds is let go. A model
idle for `IDLE` seconds is let go anyway. The next use loads it again.

A model is never torn out from under a render: its owner's lock is tried, not waited for, and a
model whose owner is busy is skipped. The renders that hold it keep their own reference, so what
is let go is the lab's hold on it, freed when the last user is done.
"""

import ctypes
import gc
import os
import threading
import time

from loadlock import LOAD

GB = 2 ** 30
MARGIN = float(os.environ.get("LAB_MEM_MARGIN_GB", 6)) * GB      # free commit kept, system-wide
GPU_MARGIN = float(os.environ.get("LAB_GPU_MARGIN_GB", 1.5)) * GB  # free memory kept on the card
PROTECT = float(os.environ.get("LAB_MEM_PROTECT_S", 45))          # a model used this recently stays
IDLE = float(os.environ.get("LAB_MEM_IDLE_S", 900))               # a model unused this long goes


def free_commit():
    """Bytes the system can still commit (Windows: RAM plus page file, less what is committed)."""
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return int(s.ullAvailPageFile)
    import psutil

    return int(psutil.virtual_memory().available)


def private_bytes():
    """This process's committed memory."""
    import psutil

    info = psutil.Process().memory_info()
    return int(getattr(info, "private", info.rss))


def free_gpu():
    try:
        import torch

        return int(torch.cuda.mem_get_info()[0]) if torch.cuda.is_available() else None
    except Exception:
        return None


def gpu_used():
    try:
        import torch

        return int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


class Entry:
    def __init__(self, key, drop, lock, ram, gpu):
        self.key, self.drop, self.lock, self.ram, self.gpu = key, drop, lock, ram, gpu
        self.loaded = self.last = time.monotonic()
        self.uses = 0


class Models:
    def __init__(self):
        self.held = {}
        self.guard = threading.RLock()
        self.log = []          # what was let go, newest last: (time, key, why)
        self._sweeper = None

    # ---- what each loader calls ----
    def loading(self, key, drop, lock=None, guess=2.0):
        """`with MODELS.loading(key, drop):` around a load: room is made first (for about `guess`
        gigabytes), the load runs under the one loading lock, and what it took is measured and kept
        with the model."""
        return _Loading(self, key, drop, lock, guess)

    def add(self, key, drop, lock=None, ram=0, gpu=0):
        with self.guard:
            self.held[key] = Entry(key, drop, lock, ram, gpu)

    def used(self, key):
        e = self.held.get(key)
        if e is not None:
            e.last = time.monotonic()
            e.uses += 1

    # ---- letting go ----
    def let_go(self, e, why):
        """Drop one model if its owner is not using it right now; True when it went."""
        if e.lock is not None and not e.lock.acquire(blocking=False):
            return False
        try:
            e.drop()
        finally:
            if e.lock is not None:
                e.lock.release()
        with self.guard:
            self.held.pop(e.key, None)
        self.log.append((time.time(), e.key, why))
        del self.log[:-50]
        print(f"  memory: let go of {e.key} ({(e.ram + e.gpu) / GB:.1f} GB, {why})", flush=True)
        return True

    def room(self, ram=0.0, gpu=0.0, protect=PROTECT):
        """Let go of the least recently used models until `ram` bytes more fit in the free commit and
        `gpu` more on the card, each with its margin. Models used in the last `protect` seconds stay."""
        freed = False
        with self.guard:
            order = sorted(self.held.values(), key=lambda e: e.last)
        for e in order:
            short_ram = free_commit() < MARGIN + ram
            g = free_gpu()
            short_gpu = g is not None and g < GPU_MARGIN + gpu
            if not (short_ram or short_gpu):
                break
            if time.monotonic() - e.last < protect:
                continue
            if short_gpu and not short_ram and e.gpu <= 0:
                continue  # the card is what is short, and this one is not on it
            freed |= self.let_go(e, "memory short")
        if freed:
            settle()
        return freed

    def sweep(self):
        """Idle models let go, and room kept: what the background thread does every few seconds."""
        now = time.monotonic()
        idle = [e for e in list(self.held.values()) if now - e.last > IDLE]
        gone = [self.let_go(e, f"idle {int(now - e.last)} s") for e in idle]
        if any(gone):
            settle()
        self.room()

    def start(self, every=5.0):
        if self._sweeper is None:
            def loop():
                while True:
                    time.sleep(every)
                    try:
                        self.sweep()
                    except Exception as e:  # the sweeper must outlive any one bad drop
                        print("  memory sweep:", e, flush=True)
            self._sweeper = threading.Thread(target=loop, name="memory", daemon=True)
            self._sweeper.start()

    def report(self):
        now = time.monotonic()
        with self.guard:
            rows = [dict(model=e.key, ram_gb=round(e.ram / GB, 2), gpu_gb=round(e.gpu / GB, 2),
                         idle_s=round(now - e.last), uses=e.uses) for e in sorted(self.held.values(), key=lambda e: -e.last)]
        g = free_gpu()
        return dict(models=rows, free_commit_gb=round(free_commit() / GB, 1), free_gpu_gb=None if g is None else round(g / GB, 1),
                    process_gb=round(private_bytes() / GB, 1), margins_gb=[MARGIN / GB, GPU_MARGIN / GB],
                    protect_s=PROTECT, idle_s=IDLE, let_go=[dict(at=t, model=k, why=w) for t, k, w in self.log[-12:]])


class _Loading:
    def __init__(self, models, key, drop, lock, guess):
        self.m, self.key, self.drop, self.lock, self.guess = models, key, drop, lock, guess

    def __enter__(self):
        LOAD.acquire()
        try:
            self.m.room(ram=self.guess * GB, gpu=self.guess * GB)
            # A load that cannot fit is refused here, as an error the page shows: let through, it
            # fails inside the checkpoint read as an access violation and takes the server with it.
            free = free_commit()
            if free < (self.guess + 1.0) * GB:
                raise MemoryError(f"not enough memory to load {self.key}: about {self.guess:.0f} GB needed, "
                                  f"{free / GB:.1f} GB free; close other programs, or let the lab's idle "
                                  f"models go (they go after {PROTECT:.0f} s unused)")
            self.ram0, self.gpu0 = private_bytes(), gpu_used()
        except BaseException:
            LOAD.release()
            raise
        return self

    def __exit__(self, kind, value, tb):
        try:
            if kind is None:
                self.m.add(self.key, self.drop, self.lock, max(0, private_bytes() - self.ram0), max(0, gpu_used() - self.gpu0))
        finally:
            LOAD.release()
        return False


def settle():
    """Memory let go is returned: Python's cycles collected and the card's cache emptied."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


MODELS = Models()
