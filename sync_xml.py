#!/usr/bin/env python3
"""
Синхронізатор XML для Adobe Premiere Pro.

Алгоритм на основі BBC audio-offset-finder (Apache 2.0):
1. Витягуємо аудіо з кожного файлу обох камер
2. Обчислюємо MFCC (Mel-Frequency Cepstral Coefficients) — стійке до шуму
   перцептуальне представлення звуку
3. Для кожного кліпу камери 2 порівнюємо його MFCC з MFCC кожного
   кліпу камери 1 (попарно) через крос-кореляцію
4. Найкращий матч (max standard_score) показує партнера та зсув
5. З цього обчислюємо абсолютну позицію на таймлайні

Натхнення: https://github.com/bbc/audio-offset-finder
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
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
    from scipy.signal import fftconvolve
    from scipy.fft import dct
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─── Аудіо ───────────────────────────────────────────────────────────────────

SAMPLE_RATE = 8000   # для MFCC достатньо
HOP_LENGTH = 128     # крок між фреймами MFCC (в семплах)
WIN_LENGTH = 256     # довжина вікна
NFFT = 512           # розмір FFT
N_MFCC = 26          # кількість MFCC коефіцієнтів (як у BBC)
N_MELS = 40          # кількість мел-фільтрів


def extract_audio(video_path, sample_rate=SAMPLE_RATE):
    """Витягує ВСЕ аудіо з файлу через ffmpeg. Повертає numpy array або None."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-ac', '1', '-ar', str(sample_rate), '-sample_fmt', 's16',
            '-loglevel', 'error',
            tmp_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)
        with wave.open(tmp_path, 'rb') as wf:
            n = wf.getnframes()
            if n == 0:
                return None
            raw = wf.readframes(n)
            return np.array(struct.unpack(f'<{n}h', raw), dtype=np.float64)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─── MFCC без librosa ────────────────────────────────────────────────────────

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(n_mels, nfft, sr):
    """Створює мел-фільтрбанк (трикутні фільтри)."""
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((nfft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, nfft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i+1], bins[i+2]
        for j in range(left, center):
            if center > left:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fb[i, j] = (right - j) / (right - center)
    return fb


def compute_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_mels=N_MELS,
                 nfft=NFFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
    """
    Обчислює MFCC без librosa.
    Повертає масив shape (n_frames, n_mfcc).
    """
    # Вікно
    window = np.hanning(win_length)

    # Розбиваємо на фрейми
    n_frames = 1 + (len(audio) - win_length) // hop_length
    if n_frames < 1:
        return np.array([]).reshape(0, n_mfcc)

    frames = np.zeros((n_frames, nfft))
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + win_length] * window
        frames[i, :win_length] = frame

    # FFT → спектр потужності
    spectrum = np.abs(np.fft.rfft(frames, n=nfft)) ** 2

    # Мел-фільтрбанк
    fb = mel_filterbank(n_mels, nfft, sr)
    mel_spec = np.dot(spectrum, fb.T)

    # Логарифм (з захистом від нуля)
    mel_spec = np.log(mel_spec + 1e-10)

    # DCT → MFCC
    mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    return mfcc


def std_mfcc(mfcc_array):
    """Z-score нормалізація по кожному коефіцієнту (як у BBC)."""
    std = np.std(mfcc_array, axis=0)
    std[std < 1e-10] = 1.0  # захист від ділення на нуль
    return (mfcc_array - np.mean(mfcc_array, axis=0)) / std


# ─── Крос-кореляція MFCC (за мотивами BBC audio-offset-finder) ──────────────

def cross_correlation_mfcc(mfcc1, mfcc2, nframes):
    """
    Крос-кореляція двох MFCC масивів.
    Повертає (correlation_array, offset_min, offset_max).

    Натхнення: BBC audio-offset-finder (Apache 2.0).
    """
    n1 = mfcc1.shape[0]
    n2 = mfcc2.shape[0]
    o_min = nframes - n2
    o_max = n1 - nframes + 1
    n = o_max - o_min
    c = np.zeros(n)

    for k in range(o_min, 0):
        cc = np.sum(np.multiply(mfcc1[:nframes], mfcc2[-k:nframes - k]), axis=0)
        c[k - o_min] = np.linalg.norm(cc)

    for k in range(0, o_max):
        cc = np.sum(np.multiply(mfcc1[k:k + nframes], mfcc2[:nframes]), axis=0)
        c[k - o_min] = np.linalg.norm(cc)

    return c, o_min, o_max


def find_offset_mfcc(mfcc1, mfcc2, max_frames=2000):
    """
    Знаходить зсув mfcc2 відносно mfcc1.
    Повертає (frame_offset, standard_score).

    frame_offset > 0: mfcc2 починається пізніше за mfcc1.
    standard_score > 10: надійний результат (за BBC).
    """
    correl_nframes = min(int(len(mfcc1) / 3), len(mfcc2), max_frames)
    if correl_nframes < 5:
        return 0, 0.0

    c, o_min, o_max = cross_correlation_mfcc(mfcc1, mfcc2, correl_nframes)

    max_idx = np.argmax(c)
    frame_offset = max_idx + o_min  # реальний зсув у фреймах

    std_c = np.std(c)
    if std_c < 1e-10:
        score = 0.0
    else:
        score = (c[max_idx] - np.mean(c)) / std_c

    return frame_offset, score


def find_best_match(tgt_clip, ref_clips, timebase, sample_rate=SAMPLE_RATE):
    """
    Для цільового кліпу знаходить найкращий матч серед референсних.
    Використовує MFCC крос-кореляцію.

    Повертає (new_start_frame, standard_score, best_ref_name) або None.
    """
    tgt_audio = tgt_clip.get('_audio')
    if tgt_audio is None:
        return None

    # Фрагмент аудіо цілі відповідно до in/out
    in_sec = tgt_clip['in'] / timebase
    out_sec = tgt_clip['out'] / timebase
    in_sample = int(in_sec * sample_rate)
    out_sample = min(int(out_sec * sample_rate), len(tgt_audio))
    tgt_fragment = tgt_audio[in_sample:out_sample]

    if len(tgt_fragment) < sample_rate * 0.5:
        return None

    # MFCC цільового фрагмента
    tgt_mfcc = compute_mfcc(tgt_fragment, sr=sample_rate)
    if len(tgt_mfcc) < 5:
        return None
    tgt_mfcc = std_mfcc(tgt_mfcc)

    best_score = -1
    best_start = 0
    best_ref_name = ""

    for ref_clip in ref_clips:
        ref_mfcc = ref_clip.get('_mfcc')
        if ref_mfcc is None:
            continue

        frame_offset, score = find_offset_mfcc(ref_mfcc, tgt_mfcc)

        if score > best_score:
            best_score = score
            # frame_offset = зсув tgt відносно ref в MFCC-фреймах
            # Переводимо у секунди
            offset_sec = frame_offset * HOP_LENGTH / sample_rate

            # Позиція на таймлайні:
            # ref_audio[0] на таймлайні = ref_start - ref_in (у фреймах таймбейзу)
            # offset_sec = де tgt_fragment[0] знаходиться відносно ref_audio[0]
            # tgt_fragment[0] = file[in] цілі
            # На таймлайні це = (ref_start - ref_in) / timebase + offset_sec
            # new_start (в секундах) = вищевказане
            # new_start (у фреймах) = ^ * timebase

            ref_file_start_sec = (ref_clip['start'] - ref_clip['in']) / timebase
            tgt_timeline_sec = ref_file_start_sec + offset_sec
            new_start_frame = int(round(tgt_timeline_sec * timebase))

            best_start = new_start_frame
            best_ref_name = ref_clip.get('name', '?')

    # Standard score < 5 — ненадійно (за BBC)
    if best_score < 3:
        return None

    return best_start, best_score, best_ref_name


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
                '_audio': None,
                '_mfcc': None,
            })

        track_info.append({'element': track, 'index': i, 'clips': clips})

    return tree, sequence, timebase, track_info


# ─── Збереження XML ──────────────────────────────────────────────────────────

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


# ─── Синхронізація ───────────────────────────────────────────────────────────

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

class SyncApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Синхронізатор XML для Premiere Pro")
        self.root.geometry("750x560")
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
        self.log = tk.Text(lf, height=8, wrap=tk.WORD, state=tk.DISABLED, font=('Menlo', 10))
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

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
            self.ref_combo.current(0)
            self.tgt_combo.current(1)
        self.btn_sync.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.DISABLED)
        self._clear_log()
        self._log(f"Відкрито: {os.path.basename(path)}")

    def start_sync(self):
        if not HAS_SCIPY:
            messagebox.showerror("Помилка", "numpy/scipy не встановлені")
            return
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
        all_clips = ref_clips + tgt_clips
        total_steps = len(all_clips) + len(ref_clips) + len(tgt_clips) + 2

        self._log(f"Референс: {len(ref_clips)} кліпів")
        self._log(f"Ціль: {len(tgt_clips)} кліпів")
        self._log(f"Витягую аудіо з {len(all_clips)} файлів...")

        def run():
            step = [0]

            def progress(msg=None):
                step[0] += 1
                pct = min(step[0] / total_steps * 100, 100)
                self.root.after(0, lambda: self.progress.config(value=pct))
                if msg:
                    self.root.after(0, lambda m=msg: self._log(m))

            try:
                # ── Крок 1: витягуємо аудіо ──
                path_cache = {}

                def get_audio(clip):
                    if not clip['pathurl']:
                        return clip, None
                    fp = pathurl_to_filepath(clip['pathurl'])
                    if fp in path_cache:
                        return clip, path_cache[fp]
                    if not os.path.exists(fp):
                        return clip, None
                    audio = extract_audio(fp)
                    path_cache[fp] = audio
                    return clip, audio

                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {pool.submit(get_audio, c): c for c in all_clips}
                    for future in as_completed(futures):
                        clip, audio = future.result()
                        clip['_audio'] = audio
                        progress()

                ref_ok = sum(1 for c in ref_clips if c['_audio'] is not None)
                tgt_ok = sum(1 for c in tgt_clips if c['_audio'] is not None)
                progress(f"Аудіо: {ref_ok}/{len(ref_clips)} реф, {tgt_ok}/{len(tgt_clips)} ціль")

                if ref_ok == 0 or tgt_ok == 0:
                    raise RuntimeError("Не вдалося витягнути аудіо з файлів")

                # ── Крок 2: обчислюємо MFCC для ПОВНИХ аудіо референсних кліпів ──
                progress("Обчислюю MFCC для референсних кліпів...")
                for rc in ref_clips:
                    if rc['_audio'] is not None:
                        full_mfcc = compute_mfcc(rc['_audio'], sr=SAMPLE_RATE)
                        if len(full_mfcc) >= 5:
                            rc['_mfcc'] = std_mfcc(full_mfcc)
                    progress()

                mfcc_ok = sum(1 for c in ref_clips if c.get('_mfcc') is not None)
                progress(f"MFCC готово: {mfcc_ok}/{len(ref_clips)} реф")

                # ── Крок 3: попарна кореляція ──
                progress("Попарна MFCC кореляція...")
                synced = 0
                skipped = 0

                for i, tgt_clip in enumerate(tgt_clips):
                    result = find_best_match(tgt_clip, ref_clips, self.timebase)
                    if result is None:
                        skipped += 1
                        progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                                 f"не знайдено матч, пропускаю")
                        continue

                    new_start, score, ref_name = result
                    old_start = tgt_clip['start']
                    delta = new_start - old_start

                    if new_start < -1000 or new_start > 500000:
                        skipped += 1
                        progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                                 f"нереалістична позиція {new_start}, пропускаю")
                        continue

                    set_clip_position(tgt_clip, new_start, self.timebase, self.sequence)
                    synced += 1

                    progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                             f"{old_start}→{new_start} (Δ{delta:+d}) "
                             f"матч={ref_name} score={score:.1f}")

                self.root.after(0, lambda: self._sync_done(synced, skipped))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self._sync_failed(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _sync_done(self, synced, skipped):
        self.progress['value'] = 100
        self.status_label.config(text="Готово!")
        self._log("─" * 40)
        self._log(f"ГОТОВО: синхронізовано {synced}, пропущено {skipped}")
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
