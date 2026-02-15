#!/usr/bin/env python3
"""
Синхронізатор XML для Adobe Premiere Pro — v5.

Алгоритм на основі аудіо-всплесків (peak fingerprinting):
1. Референс = доріжка з більшою кількістю кліпів
2. Сортуємо обидві доріжки хронологічно (по імені файлу)
3. Витягуємо аудіо (ffmpeg → WAV), знаходимо всплески (піки гучності)
4. Відбиток кліпу = список (час_піку, інтенсивність)
5. Для кожного кліпу cam2 шукаємо матч серед кліпів cam1:
   порівнюємо інтервали між піками + їх інтенсивність
6. Матч ≥ 60% → ставимо на обчислену позицію
7. Без матчу → ставимо в кінець таймлайну

Без numpy/scipy — чистий Python + ffmpeg.
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
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─── Параметри ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 8000       # для піків достатньо
ENVELOPE_WINDOW = 800    # вікно огинаючої (~100мс при 8000Hz)
PEAK_MIN_DISTANCE = 2400 # мін. відстань між піками (~300мс)
PEAK_THRESHOLD = 0.3     # поріг від максимуму огинаючої
MIN_PEAKS = 3            # мінімум піків для порівняння
INTERVAL_TOLERANCE = 0.05  # 50мс допуск для інтервалів
INTENSITY_TOLERANCE = 0.25 # 25% допуск для інтенсивності
MIN_MATCH_RATIO = 0.60   # мінімум 60% співпадінь
SEARCH_WINDOW = 10       # скільки рефів перевіряємо навколо search_start


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


# ─── Піки (всплески) ─────────────────────────────────────────────────────────

def compute_envelope(audio, window=ENVELOPE_WINDOW):
    """Огинаюча = ковзне середнє від |сигналу|. Повертає список float."""
    n = len(audio)
    if n < window:
        return [0.0] * n
    abs_audio = [abs(s) for s in audio]
    half_w = window // 2
    # Кумулятивна сума для швидкого обчислення
    cumsum = [0.0] * (n + 1)
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + abs_audio[i]
    env = [0.0] * n
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        env[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    return env


def find_peaks(audio, sample_rate=SAMPLE_RATE,
               envelope_window=ENVELOPE_WINDOW,
               min_distance=PEAK_MIN_DISTANCE,
               threshold_ratio=PEAK_THRESHOLD):
    """
    Знаходить всплески в аудіо.
    Повертає список (час_сек, інтенсивність_нормалізована).
    """
    if len(audio) < envelope_window * 2:
        return []

    env = compute_envelope(audio, envelope_window)
    max_env = max(env) if env else 0.0
    if max_env < 100:  # тиша
        return []

    threshold = max_env * threshold_ratio
    peaks = []
    last_peak_pos = -min_distance * 2  # щоб перший пік завжди пройшов

    # Шукаємо локальні максимуми вище порогу
    for i in range(1, len(env) - 1):
        if env[i] < threshold:
            continue
        if i - last_peak_pos < min_distance:
            # Якщо цей пік вищий за попередній у вікні — замінюємо
            if peaks and env[i] > peaks[-1][1] * max_env:
                peaks[-1] = (i / sample_rate, env[i] / max_env)
                last_peak_pos = i
            continue
        # Локальний максимум (вище сусідів)
        if env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            peaks.append((i / sample_rate, env[i] / max_env))
            last_peak_pos = i

    return peaks


def extract_fragment_audio(audio, in_frame, out_frame, timebase, sample_rate=SAMPLE_RATE):
    """Вирізає фрагмент аудіо за in/out фреймами таймлайну."""
    in_sec = in_frame / timebase
    out_sec = out_frame / timebase
    in_sample = int(in_sec * sample_rate)
    out_sample = min(int(out_sec * sample_rate), len(audio))
    if out_sample - in_sample < sample_rate * 0.5:
        return None
    return audio[in_sample:out_sample]


# ─── Порівняння відбитків ─────────────────────────────────────────────────────

def compute_intervals(peaks):
    """Інтервали між сусідніми піками + їх інтенсивність."""
    intervals = []
    for i in range(1, len(peaks)):
        dt = peaks[i][0] - peaks[i - 1][0]
        intensity = (peaks[i - 1][1] + peaks[i][1]) / 2.0
        intervals.append((dt, intensity))
    return intervals


def match_peaks(peaks_ref, peaks_tgt,
                interval_tol=INTERVAL_TOLERANCE,
                intensity_tol=INTENSITY_TOLERANCE,
                min_ratio=MIN_MATCH_RATIO):
    """
    Порівнює піки двох кліпів методом ковзного вікна.

    Для кожного можливого зсуву перевіряємо:
    - чи збігаються інтервали між піками (±50мс)
    - чи збігаються інтенсивності (±25%)

    Повертає (offset_sec, match_ratio, matched_count) або None.
    Offset = на скільки секунд tgt зсунути щоб він збігся з ref.
    """
    if len(peaks_ref) < MIN_PEAKS or len(peaks_tgt) < MIN_PEAKS:
        return None

    best_offset = 0.0
    best_ratio = 0.0
    best_matched = 0

    # Пробуємо всі можливі зіставлення першого піку tgt з кожним піком ref
    for ri in range(len(peaks_ref)):
        # Зсув = ref_peak_time - tgt_peak_time
        offset = peaks_ref[ri][0] - peaks_tgt[0][0]

        matched = 0
        total_compared = 0

        for ti in range(len(peaks_tgt)):
            tgt_time = peaks_tgt[ti][0] + offset
            tgt_intensity = peaks_tgt[ti][1]

            # Шукаємо найближчий пік у ref
            best_dist = float('inf')
            best_ri_match = -1
            for rj in range(len(peaks_ref)):
                dist = abs(peaks_ref[rj][0] - tgt_time)
                if dist < best_dist:
                    best_dist = dist
                    best_ri_match = rj

            total_compared += 1

            if best_dist <= interval_tol and best_ri_match >= 0:
                # Перевіряємо інтенсивність
                ref_intensity = peaks_ref[best_ri_match][1]
                if ref_intensity > 0:
                    intensity_diff = abs(tgt_intensity - ref_intensity) / ref_intensity
                else:
                    intensity_diff = abs(tgt_intensity - ref_intensity)
                if intensity_diff <= intensity_tol:
                    matched += 1

        if total_compared > 0:
            ratio = matched / total_compared
            if ratio > best_ratio:
                best_ratio = ratio
                best_offset = offset
                best_matched = matched

    if best_ratio < min_ratio or best_matched < MIN_PEAKS:
        return None

    return best_offset, best_ratio, best_matched


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

                # ── Степ 3: знаходимо піки для всіх кліпів (паралельно) ──
                progress("Шукаю піки...")

                def get_peaks_for_clip(clip, is_ref):
                    audio = audio_data.get(clip['id'])
                    if audio is None:
                        return clip, []
                    if is_ref:
                        # Для рефу — повне аудіо
                        peaks = find_peaks(audio)
                    else:
                        # Для цілі — тільки in..out
                        fragment = extract_fragment_audio(
                            audio, clip['in'], clip['out'], self.timebase)
                        if fragment is None:
                            return clip, []
                        peaks = find_peaks(fragment)
                    return clip, peaks

                peaks_data = {}
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    ref_futures = [pool.submit(get_peaks_for_clip, c, True)
                                   for c in ref_clips]
                    tgt_futures = [pool.submit(get_peaks_for_clip, c, False)
                                   for c in tgt_clips]
                    for future in as_completed(ref_futures + tgt_futures):
                        clip, peaks = future.result()
                        peaks_data[clip['id']] = peaks
                        progress()

                ref_peaks_ok = sum(1 for c in ref_clips
                                   if len(peaks_data.get(c['id'], [])) >= MIN_PEAKS)
                tgt_peaks_ok = sum(1 for c in tgt_clips
                                   if len(peaks_data.get(c['id'], [])) >= MIN_PEAKS)
                progress(f"Піки: {ref_peaks_ok}/{len(ref_clips)} реф (≥{MIN_PEAKS}), "
                         f"{tgt_peaks_ok}/{len(tgt_clips)} ціль")

                # ── Степ 4: порівнюємо піки — ковзне вікно ──
                progress("Порівнюю відбитки...")
                synced = 0
                unmatched = []
                search_start = 0

                for i, tgt_clip in enumerate(tgt_clips):
                    tgt_peaks = peaks_data.get(tgt_clip['id'], [])
                    if len(tgt_peaks) < MIN_PEAKS:
                        unmatched.append(tgt_clip)
                        progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                                 f"мало піків ({len(tgt_peaks)}) → в кінець")
                        continue

                    best_ratio = 0.0
                    best_offset_sec = 0.0
                    best_ref_name = ""
                    best_ref_idx = search_start
                    best_matched = 0

                    # Пошук: від search_start з вікном
                    window_from = max(0, search_start - SEARCH_WINDOW)
                    window_to = min(len(ref_clips), search_start + SEARCH_WINDOW + 1)

                    # Якщо вікно занадто мале — шукаємо всюди
                    if window_to - window_from < 3:
                        window_from = 0
                        window_to = len(ref_clips)

                    for ri in range(window_from, window_to):
                        ref_clip = ref_clips[ri]
                        ref_peaks = peaks_data.get(ref_clip['id'], [])
                        if len(ref_peaks) < MIN_PEAKS:
                            continue

                        result = match_peaks(ref_peaks, tgt_peaks)
                        if result is None:
                            continue

                        offset_sec, ratio, matched = result
                        if ratio > best_ratio or (ratio == best_ratio and matched > best_matched):
                            best_ratio = ratio
                            best_offset_sec = offset_sec
                            best_ref_name = ref_clip.get('name', '?')
                            best_ref_idx = ri
                            best_matched = matched

                    if best_ratio < MIN_MATCH_RATIO or best_matched < MIN_PEAKS:
                        unmatched.append(tgt_clip)
                        progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                                 f"без матчу (best {best_ratio:.0%}) → в кінець")
                        continue

                    # Обчислюємо нову позицію
                    # ref_clip починається на таймлайні = ref_start
                    # ref_audio[0] = файл[0]
                    # offset_sec = на скільки зсунути tgt щоб збігся з ref
                    # tgt_in_sec = in / timebase (початок фрагмента у файлі)
                    # new_start_sec = ref_file_start_sec + offset_sec + tgt_in_sec
                    ref_clip = ref_clips[best_ref_idx]
                    ref_file_start_sec = (ref_clip['start'] - ref_clip['in']) / self.timebase

                    tgt_in_sec = tgt_clip['in'] / self.timebase
                    new_start_sec = ref_file_start_sec + best_offset_sec
                    new_start_frame = int(round(new_start_sec * self.timebase))

                    old_start = tgt_clip['start']
                    delta = new_start_frame - old_start

                    set_clip_position(tgt_clip, new_start_frame, self.timebase,
                                      self.sequence)
                    synced += 1
                    search_start = best_ref_idx

                    progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                             f"{old_start}→{new_start_frame} (Δ{delta:+d}) "
                             f"матч={best_ref_name} {best_ratio:.0%} "
                             f"({best_matched} піків)")

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

                self.root.after(0, lambda: self._sync_done(
                    synced, len(unmatched)))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._sync_failed(str(e)))

        threading.Thread(target=run, daemon=True).start()

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
