#!/usr/bin/env python3
"""
Synthetic test suite for sync_xml.py v7 algorithm.

Tests FFT cross-correlation, envelope computation, greedy matching,
and the full pipeline (audio → envelope → FFT → offset).
"""

import sys
import os
import math
import random

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from sync_xml import (
    _fft, _ifft, _next_pow2,
    fft_cross_correlate, compute_envelope, extract_fragment_audio,
    get_file_sort_key,
    SAMPLE_RATE, ENV_WINDOW, SUBSAMPLE, MIN_CORR,
)

import cmath

PASSED = 0
FAILED = 0


def report(name, ok, detail=""):
    global PASSED, FAILED
    tag = "PASS" if ok else "FAIL"
    if ok:
        PASSED += 1
    else:
        FAILED += 1
    extra = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{extra}")


# ─── Test 1: FFT / IFFT basics ──────────────────────────────────────────────

def test_fft_basics():
    print("\n=== Test 1: FFT / IFFT basics ===")

    # Roundtrip: IFFT(FFT(x)) == x
    random.seed(42)
    data = [complex(random.random(), random.random()) for _ in range(128)]
    result = _ifft(_fft(data))
    max_err = max(abs(data[i] - result[i]) for i in range(len(data)))
    report("FFT roundtrip (128 complex)", max_err < 1e-10, f"err={max_err:.2e}")

    # Larger size
    data256 = [complex(random.random()) for _ in range(256)]
    result256 = _ifft(_fft(data256))
    max_err256 = max(abs(data256[i] - result256[i]) for i in range(len(data256)))
    report("FFT roundtrip (256 real)", max_err256 < 1e-10, f"err={max_err256:.2e}")

    # Parseval's theorem: sum|x|^2 == sum|X|^2 / N
    X = _fft(data)
    energy_time = sum(abs(d)**2 for d in data)
    energy_freq = sum(abs(x)**2 for x in X) / len(data)
    rel_err = abs(energy_time - energy_freq) / energy_time
    report("Parseval's theorem", rel_err < 1e-10, f"rel_err={rel_err:.2e}")

    # next_pow2
    report("next_pow2(1)=1", _next_pow2(1) == 1)
    report("next_pow2(5)=8", _next_pow2(5) == 8)
    report("next_pow2(1024)=1024", _next_pow2(1024) == 1024)
    report("next_pow2(1025)=2048", _next_pow2(1025) == 2048)


# ─── Test 2: fft_cross_correlate with known offsets ──────────────────────────

def test_known_offsets():
    print("\n=== Test 2: Known offsets (random envelope) ===")

    random.seed(42)
    ref = [random.random() for _ in range(500)]

    for offset_idx in [0, 50, 100, 150, 200, 250, 300, 350, 400]:
        end_idx = min(offset_idx + 100, 500)
        tgt = ref[offset_idx:end_idx]
        off, st = fft_cross_correlate(ref, tgt)
        expected_sec = offset_idx * SUBSAMPLE / SAMPLE_RATE
        err = abs(off - expected_sec)
        report(f"offset_idx={offset_idx} ({expected_sec:.1f}s)",
               err < 0.02, f"got={off:.3f}s err={err:.4f}s str={st:.4f}")


# ─── Test 3: Flat / silent signals ──────────────────────────────────────────

def test_flat_signals():
    print("\n=== Test 3: Flat / silent signals ===")

    off, st = fft_cross_correlate([0.5] * 500, [0.5] * 100)
    report("constant signals → strength=0", st == 0.0, f"str={st}")

    off, st = fft_cross_correlate([0.0] * 500, [0.0] * 100)
    report("zero signals → strength=0", st == 0.0, f"str={st}")

    # One flat, one varied
    random.seed(42)
    varied = [random.random() for _ in range(500)]
    off, st = fft_cross_correlate(varied, [0.5] * 100)
    report("varied ref + flat tgt → strength=0", st == 0.0, f"str={st}")

    off, st = fft_cross_correlate([0.5] * 500, varied[:100])
    report("flat ref + varied tgt → strength=0", st == 0.0, f"str={st}")


# ─── Test 4: Noisy matches ──────────────────────────────────────────────────

def test_noisy():
    print("\n=== Test 4: Noisy matches ===")

    random.seed(42)
    ref = [random.random() for _ in range(500)]

    for noise_level in [0.01, 0.05, 0.1, 0.2]:
        random.seed(99)
        tgt = [ref[200 + i] + random.gauss(0, noise_level) for i in range(100)]
        off, st = fft_cross_correlate(ref, tgt)
        err = abs(off - 2.0)
        report(f"noise={noise_level:.2f}", err < 0.05,
               f"offset={off:.3f}s err={err:.4f}s str={st:.4f}")


# ─── Test 5: Edge cases ─────────────────────────────────────────────────────

def test_edge_cases():
    print("\n=== Test 5: Edge cases ===")

    # Very short signals (< 5 points)
    off, st = fft_cross_correlate([1, 2, 3], [1, 2, 3])
    report("short signals (<5) → (0, 0)", st == 0.0)

    off, st = fft_cross_correlate([], [1] * 10)
    report("empty ref → (0, 0)", st == 0.0)

    off, st = fft_cross_correlate([1] * 10, [])
    report("empty tgt → (0, 0)", st == 0.0)

    # Identical signals → offset=0
    random.seed(42)
    sig = [random.random() for _ in range(300)]
    off, st = fft_cross_correlate(sig, sig)
    report("identical → offset=0", abs(off) < 0.01, f"off={off:.4f}s")
    report("identical → high strength", st > 0.9, f"str={st:.4f}")

    # Unrelated signals → low strength
    random.seed(42)
    ref = [random.random() for _ in range(500)]
    random.seed(999)
    tgt = [random.random() for _ in range(100)]
    off, st = fft_cross_correlate(ref, tgt)
    report("unrelated → low strength", st < 0.5, f"str={st:.4f}")


# ─── Test 6: Different lengths ───────────────────────────────────────────────

def test_different_lengths():
    print("\n=== Test 6: Different lengths ===")

    random.seed(42)
    ref = [random.random() for _ in range(1000)]

    # Short excerpt
    tgt = ref[500:550]
    off, st = fft_cross_correlate(ref, tgt)
    expected = 500 * SUBSAMPLE / SAMPLE_RATE
    report("short excerpt (50pts)", abs(off - expected) < 0.05,
           f"expected={expected:.1f}s got={off:.3f}s")

    # Long excerpt
    tgt2 = ref[100:400]
    off2, st2 = fft_cross_correlate(ref, tgt2)
    expected2 = 100 * SUBSAMPLE / SAMPLE_RATE
    report("long excerpt (300pts)", abs(off2 - expected2) < 0.05,
           f"expected={expected2:.1f}s got={off2:.3f}s")

    # Very long ref, short tgt
    random.seed(42)
    big_ref = [random.random() for _ in range(5000)]
    tgt3 = big_ref[2500:2600]
    off3, st3 = fft_cross_correlate(big_ref, tgt3)
    expected3 = 2500 * SUBSAMPLE / SAMPLE_RATE
    report("5000pt ref, 100pt tgt", abs(off3 - expected3) < 0.05,
           f"expected={expected3:.1f}s got={off3:.3f}s")


# ─── Test 7: Full pipeline (audio → envelope → FFT) ─────────────────────────

def test_pipeline():
    print("\n=== Test 7: Full pipeline (audio → envelope → FFT) ===")

    # Create unique random audio (10 seconds)
    random.seed(42)
    n_samples = int(10.0 * SAMPLE_RATE)
    audio_ref = [int(random.gauss(0, 10000)) for _ in range(n_samples)]

    env_ref = compute_envelope(audio_ref)
    report("envelope computed", len(env_ref) > 100, f"len={len(env_ref)}")

    # Test: slice from full envelope (no edge effects)
    for start_sec in [1.0, 3.0, 5.0, 7.0]:
        s_env = int(start_sec * SAMPLE_RATE / SUBSAMPLE)
        e_env = int((start_sec + 2.0) * SAMPLE_RATE / SUBSAMPLE)
        env_tgt = env_ref[s_env:e_env]
        off, st = fft_cross_correlate(env_ref, env_tgt)
        err = abs(off - start_sec)
        report(f"sliced env start={start_sec:.0f}s", err < 0.2,
               f"got={off:.3f}s err={err:.3f}s")

    # Test: fragment audio (as the real app does for targets)
    for start_sec in [1.0, 3.0, 5.0, 7.0]:
        s = int(start_sec * SAMPLE_RATE)
        e = int((start_sec + 2.0) * SAMPLE_RATE)
        audio_frag = audio_ref[s:e]
        env_tgt = compute_envelope(audio_frag)
        if len(env_tgt) < 5:
            report(f"fragment start={start_sec:.0f}s", False, "env too short")
            continue
        off, st = fft_cross_correlate(env_ref, env_tgt)
        err = abs(off - start_sec)
        report(f"fragment start={start_sec:.0f}s", err < 0.5,
               f"got={off:.3f}s err={err:.3f}s")


# ─── Test 8: compute_envelope ────────────────────────────────────────────────

def test_compute_envelope():
    print("\n=== Test 8: compute_envelope ===")

    # Basic properties
    random.seed(42)
    audio = [int(random.gauss(0, 5000)) for _ in range(SAMPLE_RATE * 5)]
    env = compute_envelope(audio)
    expected_len = len(range(0, len(audio), SUBSAMPLE))
    report("envelope length", len(env) == expected_len,
           f"expected={expected_len} got={len(env)}")
    report("all values positive", all(v >= 0 for v in env))
    report("no NaN/Inf", all(math.isfinite(v) for v in env))

    # Short audio (< window)
    short = [100] * (ENV_WINDOW - 1)
    env_short = compute_envelope(short)
    report("short audio → empty", len(env_short) == 0)

    # Silent audio → all zeros (or close)
    silent = [0] * (SAMPLE_RATE * 2)
    env_silent = compute_envelope(silent)
    report("silent audio → all zeros", all(v == 0 for v in env_silent))


# ─── Test 9: get_file_sort_key ───────────────────────────────────────────────

def test_file_sort_key():
    print("\n=== Test 9: get_file_sort_key ===")

    clips_mov = [
        {'name': 'P1013312.mov'},
        {'name': 'P1300434.mov'},
        {'name': 'P1013310.mov'},
    ]
    keys = [get_file_sort_key(c) for c in clips_mov]
    report("P1013312.mov → 1013312", keys[0] == 1013312)
    report("P1300434.mov → 1300434", keys[1] == 1300434)

    sorted_clips = sorted(clips_mov, key=get_file_sort_key)
    sorted_names = [c['name'] for c in sorted_clips]
    expected = ['P1013310.mov', 'P1013312.mov', 'P1300434.mov']
    report("chronological sort", sorted_names == expected, f"got {sorted_names}")

    report("no-number → string", isinstance(get_file_sort_key({'name': 'abc.mov'}), str))
    report("empty name → ''", get_file_sort_key({'name': ''}) == '')


# ─── Test 10: Greedy matching simulation ─────────────────────────────────────

def test_greedy_matching():
    print("\n=== Test 10: Greedy matching simulation ===")

    # Simulate the greedy algorithm with known data
    # 3 refs, 3 targets — each target matches best to a different ref
    random.seed(42)
    refs = [[random.random() for _ in range(500)] for _ in range(3)]
    # Target 0 = ref[2][100:200], target 1 = ref[0][200:300], target 2 = ref[1][300:400]
    tgts = [
        refs[2][100:200],
        refs[0][200:300],
        refs[1][300:400],
    ]

    # Compute all pairs
    all_pairs = []
    for ti, tgt in enumerate(tgts):
        for ri, ref in enumerate(refs):
            off, st = fft_cross_correlate(ref, tgt)
            if st >= MIN_CORR:
                all_pairs.append((st, ri, ti, off))

    # Greedy 1:1
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    assigned_ref = set()
    assigned_tgt = set()
    assignments = []

    for strength, ri, ti, off in all_pairs:
        if ri in assigned_ref or ti in assigned_tgt:
            continue
        assigned_ref.add(ri)
        assigned_tgt.add(ti)
        assignments.append((ri, ti, off, strength))

    report("3 assignments", len(assignments) == 3, f"got {len(assignments)}")

    # Check correct matching
    expected_map = {0: 2, 1: 0, 2: 1}  # tgt_idx → ref_idx
    for ri, ti, off, st in assignments:
        expected_ri = expected_map[ti]
        report(f"tgt[{ti}] → ref[{expected_ri}]", ri == expected_ri,
               f"got ref[{ri}] str={st:.4f}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("sync_xml.py v7 — Synthetic Algorithm Tests")
    print("=" * 60)
    print(f"  SAMPLE_RATE={SAMPLE_RATE}, ENV_WINDOW={ENV_WINDOW}, "
          f"SUBSAMPLE={SUBSAMPLE}, MIN_CORR={MIN_CORR}")

    try:
        test_fft_basics()
        test_known_offsets()
        test_flat_signals()
        test_noisy()
        test_edge_cases()
        test_different_lengths()
        test_pipeline()
        test_compute_envelope()
        test_file_sort_key()
        test_greedy_matching()
    except Exception:
        import traceback
        traceback.print_exc()
        FAILED += 1

    print("\n" + "=" * 60)
    total = PASSED + FAILED
    print(f"Results: {PASSED} passed, {FAILED} failed, {total} total")
    if FAILED == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"WARNING: {FAILED} test(s) failed")
    print("=" * 60)

    sys.exit(0 if FAILED == 0 else 1)
