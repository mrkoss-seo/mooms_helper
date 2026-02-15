#!/usr/bin/env python3
"""
Синхронізатор XML для Adobe Premiere Pro — v7.

Алгоритм: FFT крос-кореляція огинаючої (як у профі-інструментах).
1. Референс = доріжка з більшою кількістю кліпів
2. Сортуємо обидві доріжки хронологічно
3. Витягуємо аудіо (ffmpeg → WAV 8kHz mono)
4. Обчислюємо огинаючу (форму гучності), субсемплюємо до ~100 точок/сек
5. Для кожної пари (ref, tgt) — FFT крос-кореляція: O(N log N)
6. Жадний 1:1 матчинг: найкращі пари першими
7. Без матчу → в кінець таймлайну

Без numpy/scipy — чистий Python + ffmpeg + cmath для FFT.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import xml.etree.ElementTree as ET
import subprocess
import tempfile
import os
import re
import threading
import wave
import struct
import math
import cmath
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─── Параметри ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 8000       # 8kHz — достатньо для огинаючої
ENV_WINDOW = 1600        # вікно огинаючої (~200мс при 8000Hz)
SUBSAMPLE = 80           # субсемплювання огинаючої (кожна 80-та точка = 100 точок/сек)
MIN_CORR = 0.10          # мінімальна нормалізована крос-кореляція для матчу


# ─── Аудіо ───────────────────────────────────────────────────────────────────

def extract_audio(video_path, sample_rate=SAMPLE_RATE):
    """Витягує аудіо з файлу через ffmpeg → список int16 або None."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-ac', '1', '-ar', str(sample_rate), '-sample_fmt', 's16',
            '-loglevel', 'error', tmp_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)
        with wave.open(tmp_path, 'rb') as wf:
            n = wf.getnframes()
            if n == 0:
                return None
            raw = wf.readframes(n)
            return list(struct.unpack(f'<{n}h', raw))
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_file_sort_key(clip):
    """Ключ сортування: витягуємо числову частину з імені файлу (без розширення)."""
    name = clip.get('name', '')
    base = os.path.splitext(name)[0]
    nums = re.findall(r'\d+', base)
    if nums:
        return int(nums[-1])
    return name


# ─── Огинаюча + крос-кореляція ─────────────────────────────────────────────

def compute_envelope(audio, window=ENV_WINDOW, subsample=SUBSAMPLE):
    """
    Огинаюча = ковзне середнє від |сигналу|, субсемпльована.
    Повертає список float (сирі значення; FFT нормалізує сам).
    """
    n = len(audio)
    if n < window:
        return []
    abs_audio = [abs(s) for s in audio]
    half_w = window // 2
    # Кумулятивна сума для O(1) на точку
    cumsum = [0.0] * (n + 1)
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + abs_audio[i]
    # Субсемплюємо: беремо кожну subsample-ту точку
    env = []
    for i in range(0, n, subsample):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        env.append((cumsum[hi] - cumsum[lo]) / (hi - lo))
    return env


def extract_fragment_audio(audio, in_frame, out_frame, timebase, sample_rate=SAMPLE_RATE):
    """Вирізає фрагмент аудіо за in/out фреймами таймлайну."""
    in_sec = in_frame / timebase
    out_sec = out_frame / timebase
    in_sample = int(in_sec * sample_rate)
    out_sample = min(int(out_sec * sample_rate), len(audio))
    if out_sample - in_sample < sample_rate * 0.5:
        return None
    return audio[in_sample:out_sample]


def _fft(x):
    """Cooley-Tukey radix-2 FFT. Вхід: список complex, довжина = степінь 2."""
    N = len(x)
    if N <= 1:
        return x
    even = _fft(x[0::2])
    odd = _fft(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
           [even[k] - T[k] for k in range(N // 2)]


def _ifft(X):
    """Обернене FFT через конъюгування."""
    N = len(X)
    conj_X = [x.conjugate() for x in X]
    result = _fft(conj_X)
    return [x.conjugate() / N for x in result]


def _next_pow2(n):
    """Найменша степінь 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def fft_cross_correlate(env_ref, env_tgt, subsample=SUBSAMPLE, sample_rate=SAMPLE_RATE):
    """
    FFT крос-кореляція: знаходить зсув tgt відносно ref.

    Повертає (offset_sec, strength) де strength — нормалізована кореляція.
    Якщо обидва сигнали плоскі (std ≈ 0) → повертає (0, 0).
    """
    n_ref = len(env_ref)
    n_tgt = len(env_tgt)
    if n_tgt < 5 or n_ref < 5:
        return 0.0, 0.0

    # Нормалізуємо: zero-mean, unit-variance
    mean_a = sum(env_ref) / n_ref
    mean_b = sum(env_tgt) / n_tgt
    a = [v - mean_a for v in env_ref]
    b = [v - mean_b for v in env_tgt]

    std_a = math.sqrt(sum(v * v for v in a) / n_ref)
    std_b = math.sqrt(sum(v * v for v in b) / n_tgt)

    # Якщо один із сигналів плоский — кореляція безглузда
    if std_a < 1e-6 or std_b < 1e-6:
        return 0.0, 0.0

    a = [v / std_a for v in a]
    b = [v / std_b for v in b]

    # Pad до степені 2 >= n_ref + n_tgt - 1
    n_fft = _next_pow2(n_ref + n_tgt - 1)
    a_pad = [complex(v) for v in a] + [0j] * (n_fft - n_ref)
    b_pad = [complex(v) for v in b] + [0j] * (n_fft - n_tgt)

    # FFT cross-correlation: IFFT(FFT(a) * conj(FFT(b)))
    A = _fft(a_pad)
    B = _fft(b_pad)
    C = [aa * bb.conjugate() for aa, bb in zip(A, B)]
    corr = _ifft(C)

    # Шукаємо пік (реальна частина)
    corr_real = [c.real for c in corr]

    best_val = -1e30
    best_idx = 0
    for i in range(len(corr_real)):
        if corr_real[i] > best_val:
            best_val = corr_real[i]
            best_idx = i

    # Нормалізуємо: ділимо на min(n_ref, n_tgt) для порівняння різних пар
    strength = best_val / min(n_ref, n_tgt)

    # Конвертуємо circular index → linear offset
    if best_idx >= n_ref:
        offset_idx = best_idx - n_fft  # від'ємний зсув
    else:
        offset_idx = best_idx

    offset_sec = offset_idx * subsample / sample_rate
    return offset_sec, strength


# ─── XML ─────────────────────────────────────────────────────────────────────

def pathurl_to_filepath(pathurl):
    if pathurl.startswith('file://localhost'):
        path = pathurl[len('file://localhost'):]
    elif pathurl.startswith('file://'):
        path = pathurl[len('file://'):]
    else:
        path = pathurl
    return unquote(path)


def parse_xmeml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sequence = root.find('.//sequence')
    if sequence is None:
        raise ValueError("Не знайдено sequence")

    rate_el = sequence.find('./rate/timebase')
    timebase = int(rate_el.text) if rate_el is not None else 25

    video = sequence.find('./media/video')
    if video is None:
        raise ValueError("Не знайдено відео")

    track_info = []
    for i, track in enumerate(video.findall('track')):
        clips = []
        for ci in track.findall('clipitem'):
            name_el = ci.find('name')
            start_el = ci.find('start')
            end_el = ci.find('end')
            in_el = ci.find('in')
            out_el = ci.find('out')
            dur_el = ci.find('duration')
            file_el = ci.find('file')

            pathurl = None
            if file_el is not None:
                pu = file_el.find('pathurl')
                if pu is not None and pu.text:
                    pathurl = pu.text
                else:
                    file_id = file_el.get('id')
                    if file_id:
                        for full_file in root.iter('file'):
                            if full_file.get('id') == file_id:
                                fpu = full_file.find('pathurl')
                                if fpu is not None and fpu.text:
                                    pathurl = fpu.text
                                break

            links = []
            for link in ci.findall('link'):
                lr = link.find('linkclipref')
                mt = link.find('mediatype')
                ti_el = link.find('trackindex')
                links.append({
                    'clipref': lr.text if lr is not None else '',
                    'mediatype': mt.text if mt is not None else '',
                    'trackindex': int(ti_el.text) if ti_el is not None else 0,
                })

            start_val = int(start_el.text) if start_el is not None else 0
            end_val = int(end_el.text) if end_el is not None else 0
            in_val = int(in_el.text) if in_el is not None else 0
            out_val = int(out_el.text) if out_el is not None else 0

            clips.append({
                'element': ci,
                'id': ci.get('id', ''),
                'name': name_el.text if name_el is not None else '',
                'start': start_val,
                'end': end_val,
                'in': in_val,
                'out': out_val,
                'duration': int(dur_el.text) if dur_el is not None else 0,
                'pathurl': pathurl,
                'links': links,
            })

        track_info.append({'element': track, 'index': i, 'clips': clips})

    return tree, sequence, timebase, track_info


def save_xmeml(tree, original_path, output_path):
    with open(original_path, 'rb') as f:
        has_bom = f.read(3) == b'\xef\xbb\xbf'

    xml_str = ET.tostring(tree.getroot(), encoding='unicode')
    xml_str = re.sub(r'<(\w+)(\s+[^>]*?)\s*/>', lambda m: f'<{m.group(1)}{m.group(2)}></{m.group(1)}>', xml_str)
    xml_str = re.sub(r'<(\w+)\s*/>', lambda m: f'<{m.group(1)}></{m.group(1)}>', xml_str)

    with open(output_path, 'w', encoding='utf-8') as f:
        if has_bom:
            f.write('\ufeff')
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE xmeml>\n')
        f.write(xml_str)


def set_clip_position(clip, new_start, timebase, sequence):
    """Встановлює нову позицію кліпу на таймлайні (відео + аудіо)."""
    el = clip['element']
    clip_len = clip['out'] - clip['in']
    new_end = new_start + clip_len

    s = el.find('start')
    e = el.find('end')
    if s is not None:
        s.text = str(new_start)
    if e is not None:
        e.text = str(new_end)

    audio_section = sequence.find('./media/audio')
    if audio_section is None:
        return
    audio_tracks = audio_section.findall('track')

    for link in clip['links']:
        if link['mediatype'] != 'audio':
            continue
        clipref = link['clipref']
        tidx = link['trackindex'] - 1
        if tidx < 0 or tidx >= len(audio_tracks):
            continue
        for aci in audio_tracks[tidx].findall('clipitem'):
            if aci.get('id') == clipref:
                as_el = aci.find('start')
                ae_el = aci.find('end')
                if as_el is not None:
                    as_el.text = str(new_start)
                if ae_el is not None:
                    ae_el.text = str(new_end)
                break


# ─── GUI ─────────────────────────────────────────────────────────────────────

def _read_version():
    """Читає версію з файлу VERSION поруч зі скриптом."""
    try:
        vpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')
        with open(vpath) as f:
            return f.read().strip()
    except Exception:
        return '?'


class SyncApp:
    def __init__(self, root):
        self.root = root
        self.version = _read_version()
        self.root.title(f"Синхронізатор XML — v{self.version}")
        self.root.geometry("750x580")
        self.root.resizable(True, True)

        self.xml_path = None
        self.tree = None
        self.sequence = None
        self.timebase = 25
        self.track_info = []

        self._build_ui()

    def _build_ui(self):
        m = ttk.Frame(self.root, padding=15)
        m.pack(fill=tk.BOTH, expand=True)

        ff = ttk.LabelFrame(m, text="XML файл", padding=10)
        ff.pack(fill=tk.X, pady=(0, 10))
        self.file_label = ttk.Label(ff, text="Файл не вибрано", wraplength=550)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ff, text="Вибрати...", command=self.open_file).pack(side=tk.RIGHT)

        inf = ttk.LabelFrame(m, text="Доріжки", padding=10)
        inf.pack(fill=tk.X, pady=(0, 10))
        self.info_text = tk.Text(inf, height=4, wrap=tk.WORD, state=tk.DISABLED,
                                  font=('Menlo', 11))
        self.info_text.pack(fill=tk.X)

        pf = ttk.LabelFrame(m, text="Параметри", padding=10)
        pf.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(pf, text="Референс (камера 1):").grid(row=0, column=0, sticky=tk.W)
        self.ref_var = tk.StringVar()
        self.ref_combo = ttk.Combobox(pf, textvariable=self.ref_var, state='readonly', width=40)
        self.ref_combo.grid(row=0, column=1, padx=(10, 0))

        ttk.Label(pf, text="Синхронізувати (камера 2):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.tgt_var = tk.StringVar()
        self.tgt_combo = ttk.Combobox(pf, textvariable=self.tgt_var, state='readonly', width=40)
        self.tgt_combo.grid(row=1, column=1, padx=(10, 0), pady=(5, 0))

        self.progress = ttk.Progressbar(m, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(0, 5))
        self.status_label = ttk.Label(m, text="Готово", wraplength=700)
        self.status_label.pack(fill=tk.X)

        bf = ttk.Frame(m)
        bf.pack(fill=tk.X, pady=(5, 5))
        self.btn_sync = ttk.Button(bf, text="Синхронізувати", command=self.start_sync,
                                    state=tk.DISABLED)
        self.btn_sync.pack(side=tk.LEFT, padx=(0, 10))
        self.btn_save = ttk.Button(bf, text="Зберегти XML", command=self.save_file,
                                    state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT)

        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bf, text="Дебаг", variable=self.debug_var).pack(side=tk.RIGHT)

        lf = ttk.LabelFrame(m, text="Лог", padding=5)
        lf.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.log = tk.Text(lf, height=10, wrap=tk.WORD, state=tk.DISABLED, font=('Menlo', 10))
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        ver_frame = ttk.Frame(m)
        ver_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Label(ver_frame, text=f"v{self.version}",
                  foreground='gray', font=('Menlo', 9)).pack(side=tk.RIGHT)

    def _set_info(self, t):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert(tk.END, t)
        self.info_text.config(state=tk.DISABLED)

    def _log(self, t):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, t + '\n')
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log.config(state=tk.NORMAL)
        self.log.delete('1.0', tk.END)
        self.log.config(state=tk.DISABLED)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Вибрати XML", filetypes=[("XML", "*.xml"), ("All", "*.*")])
        if not path:
            return
        self.xml_path = path
        self.file_label.config(text=os.path.basename(path))

        try:
            self.tree, self.sequence, self.timebase, self.track_info = parse_xmeml(path)
        except Exception as e:
            messagebox.showerror("Помилка", str(e))
            return

        vt = [t for t in self.track_info if t['clips']]
        opts, info = [], [f"Timebase: {self.timebase} fps"]
        for t in vt:
            c0 = t['clips'][0]
            folder = ''
            if c0['pathurl']:
                folder = os.path.basename(os.path.dirname(pathurl_to_filepath(c0['pathurl'])))
            lbl = f"Доріжка {t['index']+1}: {len(t['clips'])} кліпів (/{folder})"
            opts.append(lbl)
            info.append(f"  {lbl}")
        self._set_info('\n'.join(info))
        self.ref_combo['values'] = opts
        self.tgt_combo['values'] = opts
        if len(opts) >= 2:
            vt_sorted = sorted(enumerate(vt), key=lambda x: len(x[1]['clips']), reverse=True)
            self.ref_combo.current(vt_sorted[0][0])
            self.tgt_combo.current(vt_sorted[1][0])
        self.btn_sync.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self._clear_log()
        self._log(f"Відкрито: {os.path.basename(path)}")

    def start_sync(self):
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            messagebox.showerror("Помилка", "ffmpeg не знайдено")
            return

        ri, ti = self.ref_combo.current(), self.tgt_combo.current()
        if ri < 0 or ti < 0 or ri == ti:
            messagebox.showwarning("Увага", "Оберіть різні доріжки")
            return

        try:
            self.tree, self.sequence, self.timebase, self.track_info = parse_xmeml(self.xml_path)
            vt = [t for t in self.track_info if t['clips']]
            ref_track, tgt_track = vt[ri], vt[ti]
        except Exception as e:
            messagebox.showerror("Помилка", str(e))
            return

        self.btn_sync.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self._clear_log()
        self.progress['value'] = 0

        ref_clips = ref_track['clips']
        tgt_clips = tgt_track['clips']

        def run():
            try:
                n_workers = max(2, os.cpu_count() or 4)
                import threading as _thr

                # ── Степ 1: сортуємо хронологічно ──
                ref_clips.sort(key=get_file_sort_key)
                tgt_clips.sort(key=get_file_sort_key)

                self.root.after(0, lambda: self._log(
                    f"Референс: {len(ref_clips)} кліпів "
                    f"({ref_clips[0]['name']}...{ref_clips[-1]['name']})"))
                self.root.after(0, lambda: self._log(
                    f"Ціль: {len(tgt_clips)} кліпів "
                    f"({tgt_clips[0]['name']}...{tgt_clips[-1]['name']})"))

                all_clips = ref_clips + tgt_clips
                total_steps = len(all_clips) + len(all_clips) + len(tgt_clips) + 3
                step = [0]
                step_lock = _thr.Lock()

                def progress(msg=None):
                    with step_lock:
                        step[0] += 1
                        pct = min(step[0] / total_steps * 100, 100)
                    self.root.after(0, lambda: self.progress.config(value=pct))
                    if msg:
                        self.root.after(0, lambda m=msg: self._log(m))

                # ── Степ 2: витягуємо аудіо (паралельно) ──
                progress("Витягую аудіо...")
                path_cache = {}
                cache_lock = _thr.Lock()

                def get_audio(clip):
                    if not clip['pathurl']:
                        return clip, None
                    fp = pathurl_to_filepath(clip['pathurl'])
                    with cache_lock:
                        if fp in path_cache:
                            return clip, path_cache[fp]
                    if not os.path.exists(fp):
                        return clip, None
                    audio = extract_audio(fp)
                    with cache_lock:
                        path_cache[fp] = audio
                    return clip, audio

                audio_data = {}
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(get_audio, c): c for c in all_clips}
                    for future in as_completed(futures):
                        clip, audio = future.result()
                        audio_data[clip['id']] = audio
                        progress()

                ref_ok = sum(1 for c in ref_clips if audio_data.get(c['id']) is not None)
                tgt_ok = sum(1 for c in tgt_clips if audio_data.get(c['id']) is not None)
                progress(f"Аудіо: {ref_ok}/{len(ref_clips)} реф, {tgt_ok}/{len(tgt_clips)} ціль")

                if ref_ok == 0 or tgt_ok == 0:
                    raise RuntimeError("Не вдалося витягнути аудіо з файлів")

                # ── Степ 3: обчислюємо огинаючі (паралельно) ──
                progress("Обчислюю огинаючі...")

                def get_envelope_for_clip(clip, is_ref):
                    audio = audio_data.get(clip['id'])
                    if audio is None:
                        return clip, []
                    # Обчислюємо огинаючу повного аудіо
                    env_full = compute_envelope(audio)
                    if is_ref:
                        return clip, env_full
                    else:
                        # Для цілі — вирізаємо з повної огинаючої (без крайових ефектів)
                        in_sec = clip['in'] / self.timebase
                        out_sec = clip['out'] / self.timebase
                        in_env = int(in_sec * SAMPLE_RATE / SUBSAMPLE)
                        out_env = int(out_sec * SAMPLE_RATE / SUBSAMPLE)
                        env_slice = env_full[in_env:out_env]
                        if len(env_slice) < 5:
                            return clip, []
                        return clip, env_slice

                env_data = {}
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    ref_futures = [pool.submit(get_envelope_for_clip, c, True)
                                   for c in ref_clips]
                    tgt_futures = [pool.submit(get_envelope_for_clip, c, False)
                                   for c in tgt_clips]
                    for future in as_completed(ref_futures + tgt_futures):
                        clip, env = future.result()
                        env_data[clip['id']] = env
                        progress()

                ref_env_ok = sum(1 for c in ref_clips
                                 if len(env_data.get(c['id'], [])) >= 5)
                tgt_env_ok = sum(1 for c in tgt_clips
                                 if len(env_data.get(c['id'], [])) >= 5)
                progress(f"Огинаючі: {ref_env_ok}/{len(ref_clips)} реф, "
                         f"{tgt_env_ok}/{len(tgt_clips)} ціль")

                # ── Степ 4: FFT крос-кореляція — повний пошук + жадний матчинг ──
                progress("FFT крос-кореляція (всі пари)...")

                # 4а. Обчислюємо ВСІ пари (ref_i, tgt_j) → (offset, strength)
                all_pairs = []  # [(strength, ri, ti, offset_sec)]

                for ti, tgt_clip in enumerate(tgt_clips):
                    env_tgt = env_data.get(tgt_clip['id'], [])
                    if len(env_tgt) < 5:
                        continue
                    for ri, ref_clip in enumerate(ref_clips):
                        env_ref = env_data.get(ref_clip['id'], [])
                        if len(env_ref) < 5:
                            continue
                        offset_sec, strength = fft_cross_correlate(
                            env_ref, env_tgt)
                        if strength >= MIN_CORR:
                            all_pairs.append((strength, ri, ti, offset_sec))
                    progress()

                progress(f"Знайдено {len(all_pairs)} кандидатів (corr≥{MIN_CORR})")

                # 4б. Жадний 1:1 матчинг: сортуємо за силою, призначаємо
                all_pairs.sort(key=lambda x: x[0], reverse=True)

                assigned_ref = set()   # ri → вже призначений
                assigned_tgt = set()   # ti → вже призначений
                match_results = {}     # tgt_id → {...}
                assignments = []       # [(ri, ti, offset_sec, strength)]

                # Прохід 1: строго 1:1
                for strength, ri, ti, offset_sec in all_pairs:
                    if ri in assigned_ref or ti in assigned_tgt:
                        continue
                    assigned_ref.add(ri)
                    assigned_tgt.add(ti)
                    assignments.append((ri, ti, offset_sec, strength))

                # Прохід 2: дозволяємо повторні рефи для ненайдених таргетів
                for strength, ri, ti, offset_sec in all_pairs:
                    if ti in assigned_tgt:
                        continue
                    assigned_tgt.add(ti)
                    assignments.append((ri, ti, offset_sec, strength))

                progress(f"Призначено {len(assignments)} матчів")

                # 4в. Застосовуємо зсуви
                synced = 0
                unmatched = []

                for ti, tgt_clip in enumerate(tgt_clips):
                    # Шукаємо призначення для цього таргета
                    match = None
                    for ri, mti, offset_sec, strength in assignments:
                        if mti == ti:
                            match = (ri, offset_sec, strength)
                            break

                    if match is None:
                        unmatched.append(tgt_clip)
                        match_results[tgt_clip['id']] = {
                            'matched': False, 'best_corr': 0.0,
                            'best_ref': '—'}
                        progress(f"  [{ti+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                                 f"без матчу → в кінець")
                        continue

                    ri, offset_sec, strength = match
                    ref_clip = ref_clips[ri]
                    ref_file_start_sec = (ref_clip['start'] - ref_clip['in']) / self.timebase

                    new_start_sec = ref_file_start_sec + offset_sec
                    new_start_frame = int(round(new_start_sec * self.timebase))

                    old_start = tgt_clip['start']
                    delta = new_start_frame - old_start

                    set_clip_position(tgt_clip, new_start_frame, self.timebase,
                                      self.sequence)
                    synced += 1

                    match_results[tgt_clip['id']] = {
                        'matched': True, 'best_corr': strength,
                        'best_ref': ref_clip.get('name', '?'),
                        'offset_sec': offset_sec,
                        'old_start': old_start, 'new_start': new_start_frame}

                    progress(f"  [{ti+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                             f"{old_start}→{new_start_frame} (Δ{delta:+d}) "
                             f"матч={ref_clip['name']} str={strength:.3f}")

                # ── Степ 5: кліпи без матчу — в кінець ──
                if unmatched:
                    ref_max_end = max(c['end'] for c in ref_clips)
                    progress(f"Ставлю {len(unmatched)} кліпів без матчу в кінець...")
                    tail_pos = ref_max_end + 100

                    for uc in unmatched:
                        clip_len = uc['out'] - uc['in']
                        set_clip_position(uc, tail_pos, self.timebase,
                                          self.sequence)
                        tail_pos += clip_len + 25

                # ── Дебаг-файл ──
                if self.debug_var.get():
                    self._write_debug(
                        ref_clips, tgt_clips, env_data, audio_data,
                        match_results, synced, unmatched)

                self.root.after(0, lambda: self._sync_done(
                    synced, len(unmatched)))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._sync_failed(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _write_debug(self, ref_clips, tgt_clips, env_data, audio_data,
                      match_results, synced_count, unmatched_clips):
        """Записує детальний дебаг-файл поруч зі скриптом."""
        import datetime
        debug_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(f"═══ ДЕБАГ СИНХРОНІЗАТОРА v{self.version} ═══\n")
                f.write(f"Дата: {datetime.datetime.now().isoformat()}\n")
                f.write(f"XML: {self.xml_path}\n")
                f.write(f"Timebase: {self.timebase} fps\n")
                f.write(f"Результат: синхронізовано {synced_count}, "
                        f"без матчу {len(unmatched_clips)}\n")
                f.write(f"\n{'─' * 70}\n")

                def env_summary(env, max_points=30):
                    """Коротке текстове представлення огинаючої."""
                    if not env:
                        return "(порожня)"
                    n = len(env)
                    avg = sum(env) / n
                    mx = max(env)
                    mn = min(env)
                    # Мініатюра (відносно max)
                    step = max(1, n // max_points)
                    mini = [env[j] for j in range(0, n, step)][:max_points]
                    if mx > 0:
                        norm = [v / mx for v in mini]
                    else:
                        norm = [0.0] * len(mini)
                    bar = ''.join('█' if v > 0.7 else '▓' if v > 0.4
                                  else '░' if v > 0.1 else ' ' for v in norm)
                    dur_sec = n * SUBSAMPLE / SAMPLE_RATE
                    return (f"{n} точок ({dur_sec:.1f}s), "
                            f"avg={avg:.1f} min={mn:.1f} max={mx:.1f}\n"
                            f"             |{bar}|")

                # ── Референсна доріжка ──
                f.write(f"\n▶ РЕФЕРЕНС ({len(ref_clips)} кліпів):\n")
                for i, c in enumerate(ref_clips):
                    audio = audio_data.get(c['id'])
                    audio_len = len(audio) if audio else 0
                    audio_dur = f"{audio_len / SAMPLE_RATE:.1f}s" if audio_len else "N/A"
                    env = env_data.get(c['id'], [])
                    fp = pathurl_to_filepath(c['pathurl']) if c['pathurl'] else 'N/A'

                    f.write(f"\n  [{i}] {c['name']}\n")
                    f.write(f"      ID: {c['id']}\n")
                    f.write(f"      Файл: {fp}\n")
                    f.write(f"      start={c['start']} end={c['end']} "
                            f"in={c['in']} out={c['out']} "
                            f"dur={c['duration']}\n")
                    f.write(f"      Аудіо: {audio_dur} "
                            f"({audio_len} семплів @ {SAMPLE_RATE}Hz)\n")
                    f.write(f"      Огинаюча: {env_summary(env)}\n")

                f.write(f"\n{'─' * 70}\n")

                # ── Цільова доріжка ──
                f.write(f"\n▶ ЦІЛЬ ({len(tgt_clips)} кліпів):\n")
                unmatched_names = {c['name'] for c in unmatched_clips}
                for i, c in enumerate(tgt_clips):
                    audio = audio_data.get(c['id'])
                    audio_len = len(audio) if audio else 0
                    audio_dur = f"{audio_len / SAMPLE_RATE:.1f}s" if audio_len else "N/A"
                    env = env_data.get(c['id'], [])
                    fp = pathurl_to_filepath(c['pathurl']) if c['pathurl'] else 'N/A'
                    status = "БЕЗ МАТЧУ" if c['name'] in unmatched_names else "OK"

                    f.write(f"\n  [{i}] {c['name']} [{status}]\n")
                    f.write(f"      ID: {c['id']}\n")
                    f.write(f"      Файл: {fp}\n")
                    f.write(f"      start={c['start']} end={c['end']} "
                            f"in={c['in']} out={c['out']} "
                            f"dur={c['duration']}\n")
                    f.write(f"      Фрагмент: "
                            f"{c['in']/self.timebase:.2f}s - "
                            f"{c['out']/self.timebase:.2f}s "
                            f"= {(c['out']-c['in'])/self.timebase:.2f}s\n")
                    f.write(f"      Аудіо файлу: {audio_dur}\n")
                    f.write(f"      Огинаюча фрагм: {env_summary(env)}\n")

                    # Результат матчу
                    mr = match_results.get(c['id'])
                    if mr:
                        if mr['matched']:
                            f.write(f"      ✓ МАТЧ: ref={mr['best_ref']} "
                                    f"corr={mr['best_corr']:.4f} "
                                    f"offset={mr['offset_sec']:.3f}s "
                                    f"pos: {mr['old_start']}→{mr['new_start']}\n")
                        else:
                            f.write(f"      ✗ Без матчу: best corr={mr['best_corr']:.4f} "
                                    f"(ref={mr['best_ref']})\n")

                f.write(f"\n{'─' * 70}\n")

                # ── Параметри алгоритму ──
                f.write(f"\n▶ ПАРАМЕТРИ:\n")
                f.write(f"  SAMPLE_RATE = {SAMPLE_RATE}\n")
                f.write(f"  ENV_WINDOW = {ENV_WINDOW}\n")
                f.write(f"  SUBSAMPLE = {SUBSAMPLE}\n")
                f.write(f"  MIN_CORR = {MIN_CORR}\n")
                f.write(f"  Алгоритм: FFT крос-кореляція + жадний 1:1 матчинг\n")

                f.write(f"\n{'═' * 70}\n")
                f.write(f"Файл: {debug_path}\n")

            self.root.after(0, lambda: self._log(
                f"Дебаг записано: {os.path.basename(debug_path)}"))
        except Exception as e:
            self.root.after(0, lambda: self._log(f"Помилка дебагу: {e}"))

    def _sync_done(self, synced, unmatched):
        self.progress['value'] = 100
        self.status_label.config(text="Готово!")
        self._log("─" * 40)
        self._log(f"ГОТОВО: синхронізовано {synced}, без матчу {unmatched}")
        if unmatched > 0:
            self._log(f"  (кліпи без матчу поставлені в кінець таймлайну)")
        self._log("Натисніть 'Зберегти XML'")
        self.btn_save.config(state=tk.NORMAL)
        self.btn_sync.config(state=tk.NORMAL)

    def _sync_failed(self, msg):
        self.progress['value'] = 0
        self.status_label.config(text="Помилка!")
        self._log(f"ПОМИЛКА: {msg}")
        self.btn_sync.config(state=tk.NORMAL)
        messagebox.showerror("Помилка", msg)

    def save_file(self):
        if self.tree is None:
            return
        default = os.path.splitext(os.path.basename(self.xml_path))[0] + "_synced.xml"
        path = filedialog.asksaveasfilename(
            title="Зберегти", defaultextension=".xml", initialfile=default,
            filetypes=[("XML", "*.xml")])
        if not path:
            return
        try:
            save_xmeml(self.tree, self.xml_path, path)
            self._log(f"Збережено: {path}")
            messagebox.showinfo("Готово", f"Збережено:\n{path}")
        except Exception as e:
            messagebox.showerror("Помилка", str(e))


if __name__ == '__main__':
    root = tk.Tk()
    SyncApp(root)
    root.mainloop()
