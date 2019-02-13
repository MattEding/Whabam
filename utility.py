from collections import namedtuple
from pathlib import Path
import subprocess

import librosa as lr
import numpy as np
import scipy as sp


directory = Path.cwd() / 'tracks'

def get_audio_tracks():
    """Return paths to all the audio tracks."""

    return sorted(track for track in directory.iterdir() if get_duration(track) >= 30)


def find_file(query, path=directory):
    """Return a path to file in path if query is in filename.
    Raise FileNotFoundError if it does not exist."""

    file = sorted(p for p in path.iterdir() if query in p.name)[0]
    if not file.exists():
        raise FileNotFoundError
    return file


def get_duration(file):
    """Return the duration of an audio file"""

    args=("ffprobe", "-show_entries", "format=duration", "-i", file)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    duration = output.split()[1].split(b'=')[1]
    return float(duration)


def load_audio_segments(file, sec=30):
    """Return audio_segments list (without sr) and track name from file of duration sec"""

    duration = get_duration(file)
    segments = int(duration) // sec
    audio_segments = [lr.load(file, offset=sec*i, duration=sec)[0] for i in range(segments)]
    name = file.stem.split('-')[-1].replace('_', ' ').title()
    return audio_segments, name


def get_spectrograms(audio):
    """Return namedtuple Spectrograms with STFT magnitude, Mel, CQT, Chroma, MFCC"""

    Spectrograms = namedtuple('Spectrograms', 'mag mel cqt chroma mfcc')
    stft = lr.stft(audio)
    mag = abs(stft)
    mel = lr.feature.melspectrogram(audio)
    cqt = lr.cqt(audio)
    chroma = lr.feature.chroma_cqt(audio)
    mfcc = lr.feature.mfcc(audio)
    return Spectrograms(mag, mel, cqt, chroma, mfcc)


def beat_synchronous_chroma(audio, semi_bins=3):
    """Return chroma_sync and beat_time"""

    chroma = chroma_enchanced(audio, semi_bins)
    tempo, beat_freq = lr.beat.beat_track(audio, trim=False)
    beat_freq = lr.util.fix_frames(beat_freq, x_max=chroma.shape[1])
    chroma_sync = lr.util.sync(chroma, beat_freq, aggregate=np.median)
    beat_time = lr.frames_to_time(beat_freq)
    return chroma_sync, beat_time


def chroma_enchanced(audio, semi_bins=3):
    """Return smoothed chroma"""

    harmonic = lr.effects.harmonic(audio)
    chroma_oversample = lr.feature.chroma_cqt(harmonic, bins_per_octave=12*semi_bins)
    chroma_filter = np.minimum(chroma_oversample,
                               lr.decompose.nn_filter(chroma_oversample,
                                                      aggregate=np.median,
                                                      metric='cosine'))
    chroma_smooth = sp.ndimage.median_filter(chroma_filter, size=(1, 9))
    return chroma_smooth


def label_formatter(axis):
    """Formats the labels of axis"""

    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)

    if axis.is_first_col():
        axis.yaxis.set_visible(True)
    if axis.is_last_row():
        axis.xaxis.set_visible(True)


def mode(a, axis=0):
    """Return the mode of an array"""

    return sp.stats.mode(a, axis=axis)[0].T[0]


# Note: np.max is alias for np.amax
_aggs = [np.mean, np.median, mode, np.amax, np.amin, np.std, np.sum, np.prod]
Aggregations = namedtuple('Aggregations', [agg.__name__ for agg in _aggs])

def get_aggregations(arr, axis=None):
    """Return namedtuple Aggreations with mean, median, mode, amax, amin, std, sum, prod"""

    return Aggregations(*[agg(arr, axis=axis) for agg in _aggs])


def dynamic_tempo_estimation(audio):
    """Return tempogram, dynamic_tempo, and tempo"""

    onset_env = lr.onset.onset_strength(audio)
    tempo = lr.beat.tempo(onset_envelope=onset_env)
    dynamic_tempo = lr.beat.tempo(onset_envelope=onset_env, aggregate=None)
    tempogram = lr.feature.tempogram(onset_envelope=onset_env)
    return tempogram, dynamic_tempo, tempo


def vocal_separation(audio, margin_instr=2, margin_vocal=10):
    """Return separation of foreground (vocal) and background (instrument) signals"""

    mag = abs(lr.stft(audio))
    fltr = lr.decompose.nn_filter(mag,
                                  aggregate=np.median,
                                  metric='cosine',
                                  width=int(lr.time_to_frames(2)))
    fltr = np.minimum(mag, fltr)

    mask_instr = lr.util.softmask(fltr, margin_instr * (mag - fltr), power=2)
    mask_vocal = lr.util.softmask(mag - fltr, margin_vocal * fltr, power=2)

    foreground = mask_vocal * mag
    background = mask_instr * mag
    return foreground, background
