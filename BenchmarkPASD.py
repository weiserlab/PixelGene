"""
Timing benchmark for PASD inference on the cluster.

Two ways to use it:

MODE A -- wrap your existing command (no code changes to test_pasd_NVDIA.py):
    python benchmark_pasd.py cmd --repeats 3 -- \
        python test_pasd_NVDIA.py --control_type grayscale \
        --image_path inputs/strawberry.png --output_dir outputs/bench

    Times the whole process (includes model loading every repeat).

MODE B -- import into your test script for fine-grained timing
          (separates model-load time from per-image inference time,
          which is what you usually want to report):

    from benchmark_pasd import GPUTimer, Benchmark

    bench = Benchmark(repeats=3, warmup=1)
    ...
    with GPUTimer() as t_load:
        pipeline = load_pasd_pipeline(args)      # your existing loading code
    print(f'model load: {t_load.elapsed:.2f}s')

    for rep in bench:                            # runs warmup + 3 timed repeats
        with rep:
            image = pipeline(prompt, image=lr_image, ...).images[0]
    bench.report('single image inference')

Notes for a fair measurement on a shared school server:
  * run on a COMPUTE node via the scheduler, not the login node
  * request an exclusive GPU if possible; a shared GPU makes timings noisy
  * always include >=1 warmup run (CUDA kernels compile / caches fill on
    the first call, which can be 2-5x slower than steady state)
  * torch.cuda.synchronize() before reading the clock -- GPU calls are
    asynchronous and unsynchronized wall-clock timings are meaningless
"""

import argparse
import statistics
import subprocess
import sys
import time

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ----------------------------------------------------------------------------
# GPU-aware timer
# ----------------------------------------------------------------------------

class GPUTimer:
    """Context manager measuring wall time with CUDA synchronization,
    so asynchronous GPU work is fully included."""

    def __enter__(self):
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0
        return False


# ----------------------------------------------------------------------------
# Repeat / average harness (MODE B)
# ----------------------------------------------------------------------------

class Benchmark:
    """Iterate to get `warmup` untimed runs followed by `repeats` timed runs.

        bench = Benchmark(repeats=3, warmup=1)
        for rep in bench:
            with rep:
                <the thing to time>
        bench.report('pasd inference')
    """

    class _Rep:
        def __init__(self, parent, timed):
            self.parent, self.timed = parent, timed

        def __enter__(self):
            self.timer = GPUTimer().__enter__()
            return self

        def __exit__(self, *exc):
            self.timer.__exit__(*exc)
            if self.timed and exc[0] is None:
                self.parent.times.append(self.timer.elapsed)
            return False

    def __init__(self, repeats=3, warmup=1):
        self.repeats, self.warmup = repeats, warmup
        self.times = []

    def __iter__(self):
        for i in range(self.warmup):
            print(f'[bench] warmup {i + 1}/{self.warmup} (not timed)')
            yield self._Rep(self, timed=False)
        for i in range(self.repeats):
            print(f'[bench] repeat {i + 1}/{self.repeats}')
            yield self._Rep(self, timed=True)

    def report(self, label='run'):
        if not self.times:
            print('[bench] no timed runs recorded')
            return
        mean = statistics.mean(self.times)
        std = statistics.stdev(self.times) if len(self.times) > 1 else 0.0
        print(f'\n===== {label} =====')
        for i, t in enumerate(self.times, 1):
            print(f'  run {i}: {t:8.2f} s')
        print(f'  mean : {mean:8.2f} s')
        print(f'  std  : {std:8.2f} s')
        print(f'  min  : {min(self.times):8.2f} s')
        print(f'  max  : {max(self.times):8.2f} s')
        if _HAS_TORCH and torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f'  peak GPU mem: {mem:.2f} GB')
        return mean, std


# ----------------------------------------------------------------------------
# Per-stage breakdown (encode / diffusion / decode / postprocess ...)
# ----------------------------------------------------------------------------

class StageTimer:
    """Accumulates named stage timings across repeats.

        stages = StageTimer()
        for rep in bench:
            with rep:
                with stages('encode'):
                    ...
                with stages('diffusion'):
                    ...
                with stages('vae_decode'):
                    ...
                with stages('yuv_merge'):
                    ...
        stages.report()
    """

    def __init__(self):
        self.records = {}   # name -> list of seconds

    class _Stage:
        def __init__(self, parent, name):
            self.parent, self.name = parent, name

        def __enter__(self):
            self.timer = GPUTimer().__enter__()
            return self

        def __exit__(self, *exc):
            self.timer.__exit__(*exc)
            if exc[0] is None:
                self.parent.records.setdefault(self.name, []).append(
                    self.timer.elapsed)
            return False

    def __call__(self, name):
        return self._Stage(self, name)

    def report(self):
        if not self.records:
            print('[stages] nothing recorded')
            return
        total_mean = sum(statistics.mean(v) for v in self.records.values())
        print('\n===== per-stage breakdown (mean over repeats) =====')
        print(f'  {"stage":<16}{"mean(s)":>10}{"std(s)":>10}{"share":>8}')
        for name, vals in self.records.items():
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f'  {name:<16}{m:>10.2f}{s:>10.2f}{m / total_mean:>7.1%}')
        print(f'  {"TOTAL":<16}{total_mean:>10.2f}')


# ----------------------------------------------------------------------------
# Per-denoising-step timestamps (diffusers callback)
# ----------------------------------------------------------------------------

class StepTimer:
    """Timestamps every denoising step via the diffusers pipeline callback.

    Newer diffusers (callback_on_step_end API):
        steps = StepTimer()
        steps.start()
        image = pipeline(..., callback_on_step_end=steps.on_step_end).images[0]
        steps.report()

    Older diffusers (callback= API, which PASD forks often use):
        steps = StepTimer()
        steps.start()
        image = pipeline(..., callback=steps.legacy_callback,
                         callback_steps=1).images[0]
        steps.report()
    """

    def __init__(self):
        self.step_times = []       # list of lists (one list per run)
        self._current = None
        self._last = None

    def start(self):
        """Call immediately before invoking the pipeline."""
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._current = []
        self._last = time.perf_counter()

    def _tick(self):
        if _HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        self._current.append(now - self._last)
        self._last = now

    # new-style diffusers callback
    def on_step_end(self, pipe, step, timestep, callback_kwargs):
        self._tick()
        return callback_kwargs

    # old-style diffusers callback (callback=, callback_steps=1)
    def legacy_callback(self, step, timestep, latents):
        self._tick()

    def end_run(self):
        """Call after the pipeline returns to close out this run."""
        if self._current:
            self.step_times.append(self._current)
        self._current = None

    def report(self, show_each_step=True):
        if self._current:            # user forgot end_run()
            self.end_run()
        if not self.step_times:
            print('[steps] nothing recorded')
            return
        n_steps = min(len(r) for r in self.step_times)
        print(f'\n===== per-step timing over {len(self.step_times)} run(s) =====')
        if show_each_step:
            print(f'  {"step":<6}{"mean(s)":>10}{"std(s)":>10}')
            for i in range(n_steps):
                vals = [r[i] for r in self.step_times]
                m = statistics.mean(vals)
                s = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f'  {i:<6}{m:>10.3f}{s:>10.3f}')
        per_run_totals = [sum(r) for r in self.step_times]
        all_steps = [t for r in self.step_times for t in r]
        print(f'  steps/run       : {n_steps}')
        print(f'  mean step time  : {statistics.mean(all_steps):.3f} s')
        print(f'  mean loop total : {statistics.mean(per_run_totals):.2f} s')


# ----------------------------------------------------------------------------
# Whole-command benchmark (MODE A)
# ----------------------------------------------------------------------------

def benchmark_command(cmd, repeats=3, warmup=0):
    times = []
    total = warmup + repeats
    for i in range(total):
        timed = i >= warmup
        tag = 'warmup' if not timed else f'repeat {i - warmup + 1}/{repeats}'
        print(f'[bench] {tag}: {" ".join(cmd)}')
        t0 = time.perf_counter()
        ret = subprocess.run(cmd)
        dt = time.perf_counter() - t0
        if ret.returncode != 0:
            print(f'[bench] command failed (exit {ret.returncode}), aborting')
            sys.exit(ret.returncode)
        print(f'[bench] {tag} took {dt:.2f} s')
        if timed:
            times.append(dt)

    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    print('\n===== whole-command timing (includes model load) =====')
    for i, t in enumerate(times, 1):
        print(f'  run {i}: {t:8.2f} s')
    print(f'  mean : {mean:8.2f} s')
    print(f'  std  : {std:8.2f} s')


def main():
    ap = argparse.ArgumentParser(
        description='Benchmark PASD inference time (mean over repeats).')
    sub = ap.add_subparsers(dest='mode', required=True)

    cp = sub.add_parser('cmd', help='time a whole command, repeated')
    cp.add_argument('--repeats', type=int, default=3)
    cp.add_argument('--warmup', type=int, default=0,
                    help='untimed runs before timing (default 0 for whole-'
                         'command mode since each run reloads the model)')
    cp.add_argument('command', nargs=argparse.REMAINDER,
                    help='the command to run, after a bare --')

    args = ap.parse_args()
    if args.mode == 'cmd':
        cmd = args.command
        if cmd and cmd[0] == '--':
            cmd = cmd[1:]
        if not cmd:
            ap.error('no command given; usage: benchmark_pasd.py cmd -- <cmd>')
        benchmark_command(cmd, args.repeats, args.warmup)


if __name__ == '__main__':
    main()