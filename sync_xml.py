#!/usr/bin/env python3
"""
Синхронізатор XML для Adobe Premiere Pro — v4.

Алгоритм:
1. Референс = доріжка з більшою кількістю кліпів
2. Сортуємо обидві доріжки хронологічно (по імені файлу)
3. Витягуємо аудіо, обчислюємо MFCC
4. Ковзне вікно: для кожного кліпу cam2 шукаємо матч серед кліпів cam1,
   починаючи від позиції попереднього матчу (не з початку)
5. Матч з score >= 9 → ставимо на обчислену позицію
6. Без матчу → ставимо в кінець таймлайну (мама вирішить)

MFCC-кореляція натхненна BBC audio-offset-finder (Apache 2.0).
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
    from scipy.fft import dct
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─── Параметри MFCC ──────────────────────────────────────────────────────────

SAMPLE_RATE = 8000
HOP_LENGTH = 128
WIN_LENGTH = 256
NFFT = 512
N_MFCC = 26
N_MELS = 40
MIN_SCORE = 9       # мінімальний standard score для матчу
WINDOW_MARGIN = 5   # ковзне вікно: скільки кліпів назад дозволяємо


# ─── Аудіо ───────────────────────────────────────────────────────────────────

def extract_audio(video_path, sample_rate=SAMPLE_RATE):
    """Витягує аудіо з файлу через ffmpeg → numpy array або None."""
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
            return np.array(struct.unpack(f'<{n}h', raw), dtype=np.float64)
    except Exception:
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def get_file_sort_key(clip):
    """Ключ сортування: витягуємо числову частину з імені файлу (без розширення)."""
    name = clip.get('name', '')
    # Прибираємо розширення: P1013312.MP4 → P1013312
    base = os.path.splitext(name)[0]
    # P1013312 → 1013312
    nums = re.findall(r'\d+', base)
    if nums:
        return int(nums[-1])
    return name


# ─── MFCC без librosa ────────────────────────────────────────────────────────

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(n_mels, nfft, sr):
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

_mel_fb_cache = {}

def compute_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_mels=N_MELS,
                 nfft=NFFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
    """MFCC без librosa. Повертає (n_frames, n_mfcc)."""
    window = np.hanning(win_length)
    n_frames = 1 + (len(audio) - win_length) // hop_length
    if n_frames < 1:
        return np.array([]).reshape(0, n_mfcc)

    frames = np.zeros((n_frames, nfft))
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + win_length] * window
        frames[i, :win_length] = frame

    spectrum = np.abs(np.fft.rfft(frames, n=nfft)) ** 2

    key = (n_mels, nfft, sr)
    if key not in _mel_fb_cache:
        _mel_fb_cache[key] = mel_filterbank(n_mels, nfft, sr)
    fb = _mel_fb_cache[key]

    mel_spec = np.dot(spectrum, fb.T)
    mel_spec = np.log(mel_spec + 1e-10)
    mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return mfcc


def std_mfcc(mfcc_array):
    """Z-score нормалізація (як у BBC audio-offset-finder)."""
    std = np.std(mfcc_array, axis=0)
    std[std < 1e-10] = 1.0
    return (mfcc_array - np.mean(mfcc_array, axis=0)) / std


# ─── Крос-кореляція MFCC ─────────────────────────────────────────────────────

def cross_correlation_mfcc(mfcc1, mfcc2, nframes):
    """Крос-кореляція (натхнення: BBC audio-offset-finder, Apache 2.0)."""
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
    """Зсув mfcc2 відносно mfcc1. Повертає (frame_offset, standard_score)."""
    correl_nframes = min(int(len(mfcc1) / 3), len(mfcc2), max_frames)
    if correl_nframes < 5:
        return 0, 0.0
    c, o_min, o_max = cross_correlation_mfcc(mfcc1, mfcc2, correl_nframes)
    max_idx = np.argmax(c)
    frame_offset = max_idx + o_min
    std_c = np.std(c)
    if std_c < 1e-10:
        return 0, 0.0
    score = (c[max_idx] - np.mean(c)) / std_c
    return frame_offset, score


def match_clip_in_window(tgt_clip, ref_clips, search_start, timebase,
                         sample_rate=SAMPLE_RATE):
    """
    Шукає матч для tgt_clip серед ref_clips[search_start:].
    Ковзне вікно: не шукаємо раніше search_start.

    Повертає (new_start_frame, score, ref_name, matched_ref_index) або None.
    """
    tgt_audio = tgt_clip.get('_audio')
    if tgt_audio is None:
        return None

    in_sec = tgt_clip['in'] / timebase
    out_sec = tgt_clip['out'] / timebase
    in_sample = int(in_sec * sample_rate)
    out_sample = min(int(out_sec * sample_rate), len(tgt_audio))
    tgt_fragment = tgt_audio[in_sample:out_sample]

    if len(tgt_fragment) < sample_rate * 0.5:
        return None

    tgt_mfcc = compute_mfcc(tgt_fragment, sr=sample_rate)
    if len(tgt_mfcc) < 5:
        return None
    tgt_mfcc = std_mfcc(tgt_mfcc)

    best_score = -1
    best_start = 0
    best_ref_name = ""
    best_ref_idx = search_start

    # Шукаємо від search_start (з невеликим запасом назад)
    window_from = max(0, search_start - WINDOW_MARGIN)

    for ref_idx in range(window_from, len(ref_clips)):
        ref_clip = ref_clips[ref_idx]
        ref_mfcc = ref_clip.get('_mfcc')
        if ref_mfcc is None:
            continue

        frame_offset, score = find_offset_mfcc(ref_mfcc, tgt_mfcc)

        if score > best_score:
            best_score = score
            offset_sec = frame_offset * HOP_LENGTH / sample_rate
            ref_file_start_sec = (ref_clip['start'] - ref_clip['in']) / timebase
            tgt_timeline_sec = ref_file_start_sec + offset_sec
            new_start_frame = int(round(tgt_timeline_sec * timebase))
            best_start = new_start_frame
            best_ref_name = ref_clip.get('name', '?')
            best_ref_idx = ref_idx

    if best_score < MIN_SCORE:
        return None

    return best_start, best_score, best_ref_name, best_ref_idx


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


def remove_clip_from_track(clip, sequence):
    """Видаляє кліп з відео- та аудіо-доріжок."""
    el = clip['element']
    parent = None
    # Знаходимо батьківський track
    for track in sequence.find('./media/video').findall('track'):
        if el in list(track):
            parent = track
            break
    if parent is not None:
        parent.remove(el)

    # Видаляємо пов'язані аудіо
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
        for aci in list(audio_tracks[tidx]):
            if aci.get('id') == clipref:
                audio_tracks[tidx].remove(aci)
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

        # ── Версія внизу ──
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
            # Автовибір: більше кліпів = референс
            vt_sorted = sorted(enumerate(vt), key=lambda x: len(x[1]['clips']), reverse=True)
            self.ref_combo.current(vt_sorted[0][0])
            self.tgt_combo.current(vt_sorted[1][0])
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

        def run():
            try:
                n_workers = max(2, os.cpu_count() or 4)

                # ── Степ 1-2: сортуємо хронологічно ──
                ref_clips.sort(key=get_file_sort_key)
                tgt_clips.sort(key=get_file_sort_key)

                self.root.after(0, lambda: self._log(
                    f"Референс: {len(ref_clips)} кліпів "
                    f"({ref_clips[0]['name']}...{ref_clips[-1]['name']})"))
                self.root.after(0, lambda: self._log(
                    f"Ціль: {len(tgt_clips)} кліпів "
                    f"({tgt_clips[0]['name']}...{tgt_clips[-1]['name']})"))
                self.root.after(0, lambda: self._log(
                    f"Потоки: {n_workers}"))

                all_clips = ref_clips + tgt_clips
                # аудіо + mfcc(all) + матчі(tgt*ref) + вікно + фіналізація
                total_steps = (len(all_clips) + len(all_clips)
                               + len(tgt_clips) * len(ref_clips)
                               + len(tgt_clips) + 3)
                step = [0]
                import threading as _thr
                step_lock = _thr.Lock()

                def progress(msg=None):
                    with step_lock:
                        step[0] += 1
                        pct = min(step[0] / total_steps * 100, 100)
                    self.root.after(0, lambda: self.progress.config(value=pct))
                    if msg:
                        self.root.after(0, lambda m=msg: self._log(m))

                # ── Степ 3: витягуємо аудіо (паралельно, ffmpeg) ──
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

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
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

                # ── Степ 4: MFCC для ВСІХ кліпів (паралельно) ──
                progress("MFCC (паралельно)...")

                def compute_clip_mfcc(clip, is_ref):
                    """Обчислює MFCC для кліпу. Для ref — повне аудіо,
                    для tgt — тільки in..out фрагмент."""
                    audio = clip.get('_audio')
                    if audio is None:
                        return clip, None
                    if is_ref:
                        fragment = audio
                    else:
                        in_sec = clip['in'] / self.timebase
                        out_sec = clip['out'] / self.timebase
                        in_sample = int(in_sec * SAMPLE_RATE)
                        out_sample = min(int(out_sec * SAMPLE_RATE), len(audio))
                        fragment = audio[in_sample:out_sample]
                        if len(fragment) < SAMPLE_RATE * 0.5:
                            return clip, None
                    mfcc = compute_mfcc(fragment, sr=SAMPLE_RATE)
                    if len(mfcc) < 5:
                        return clip, None
                    return clip, std_mfcc(mfcc)

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    # Ref кліпи — повне аудіо
                    ref_futures = {pool.submit(compute_clip_mfcc, c, True): c
                                   for c in ref_clips}
                    # Tgt кліпи — in..out фрагмент
                    tgt_futures = {pool.submit(compute_clip_mfcc, c, False): c
                                   for c in tgt_clips}

                    for future in as_completed(list(ref_futures) + list(tgt_futures)):
                        clip, mfcc = future.result()
                        clip['_mfcc'] = mfcc
                        progress()

                ref_mfcc_ok = sum(1 for c in ref_clips if c['_mfcc'] is not None)
                tgt_mfcc_ok = sum(1 for c in tgt_clips if c['_mfcc'] is not None)
                progress(f"MFCC: {ref_mfcc_ok}/{len(ref_clips)} реф, "
                         f"{tgt_mfcc_ok}/{len(tgt_clips)} ціль")

                # ── Степ 5: попарна кореляція (паралельно) ──
                # Для кожного tgt[i] × ref[j] знаходимо (offset, score)
                progress("Попарна кореляція (паралельно)...")

                # match_matrix[i][j] = (new_start_frame, score) або None
                match_matrix = [[None] * len(ref_clips)
                                for _ in range(len(tgt_clips))]

                def compute_pair(ti, ri):
                    """Кореляція tgt[ti] з ref[ri]. Повертає (ti, ri, result)."""
                    tgt_mfcc = tgt_clips[ti].get('_mfcc')
                    ref_mfcc = ref_clips[ri].get('_mfcc')
                    if tgt_mfcc is None or ref_mfcc is None:
                        return ti, ri, None

                    frame_offset, score = find_offset_mfcc(ref_mfcc, tgt_mfcc)
                    if score < MIN_SCORE:
                        return ti, ri, None

                    offset_sec = frame_offset * HOP_LENGTH / SAMPLE_RATE
                    ref_clip = ref_clips[ri]
                    ref_file_start_sec = (ref_clip['start'] - ref_clip['in']) / self.timebase
                    tgt_timeline_sec = ref_file_start_sec + offset_sec
                    new_start_frame = int(round(tgt_timeline_sec * self.timebase))
                    return ti, ri, (new_start_frame, score)

                pairs = [(ti, ri) for ti in range(len(tgt_clips))
                         for ri in range(len(ref_clips))]

                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = [pool.submit(compute_pair, ti, ri) for ti, ri in pairs]
                    for future in as_completed(futures):
                        ti, ri, result = future.result()
                        match_matrix[ti][ri] = result
                        progress()

                # ── Степ 6: ковзне вікно (миттєво — дані вже готові) ──
                progress("Ковзне вікно: вибір найкращих матчів...")
                synced = 0
                unmatched = []
                search_start = 0

                for i, tgt_clip in enumerate(tgt_clips):
                    window_from = max(0, search_start - WINDOW_MARGIN)
                    best_score = -1
                    best_start = 0
                    best_ref_name = ""
                    best_ref_idx = search_start

                    for ri in range(window_from, len(ref_clips)):
                        result = match_matrix[i][ri]
                        if result is None:
                            continue
                        new_start, score = result
                        if score > best_score:
                            best_score = score
                            best_start = new_start
                            best_ref_name = ref_clips[ri].get('name', '?')
                            best_ref_idx = ri

                    if best_score < MIN_SCORE:
                        unmatched.append(tgt_clip)
                        progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                                 f"без матчу → в кінець")
                        continue

                    # М'яка перевірка
                    if best_start < -50000:
                        unmatched.append(tgt_clip)
                        progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                                 f"аномальна позиція ({best_start}) → в кінець")
                        continue

                    old_start = tgt_clip['start']
                    delta = best_start - old_start

                    set_clip_position(tgt_clip, best_start, self.timebase,
                                      self.sequence)
                    synced += 1
                    search_start = best_ref_idx

                    progress(f"  [{i+1}/{len(tgt_clips)}] {tgt_clip['name']}: "
                             f"{old_start}→{best_start} (Δ{delta:+d}) "
                             f"матч={best_ref_name} score={best_score:.1f}")

                # ── Степ 7: кліпи без матчу — ставимо в кінець ──
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
