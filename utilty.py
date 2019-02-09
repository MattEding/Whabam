from collections import namedtuple
from pathlib import Path
import subprocess

import librosa as lr
import numpy as np
import pandas as pd
import scipy as sp


directory = Path.cwd() / 'tracks'

def get_audio_tracks():
    """Return paths to all the audio tracks."""
    return sorted(directory.iterdir())


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


def beat_synchronous_chroma(audio):
    """Return chroma_sync and beat_time"""

    chroma = lr.feature.chroma_cqt(audio)
    tempo, beat_freq = lr.beat.beat_track(audio, trim=False)
    beat_freq = lr.util.fix_frames(beat_freq, x_max=chroma.shape[1])
    chroma_sync = lr.util.sync(chroma, beat_freq, aggregate=np.median)
    beat_time = lr.frames_to_time(beat_freq)
    return chroma_sync, beat_time


def beat_synchronous_chroma_enhanced(audio, semi_bins=3):
    """Return chroma_sync and beat_time"""

    harm = lr.effects.harmonic(audio)
    chroma_oversample = lr.feature.chroma_cqt(harm, bins_per_octave=12*semi_bins)
    chroma_filter = np.minimum(chroma_oversample,
                               lr.decompose.nn_filter(chroma_oversample,
                                                      aggregate=np.median,
                                                      metric='cosine'))
    chroma_smooth = sp.ndimage.median_filter(chroma_filter, size=(1, 9))

    tempo, beat_freq = lr.beat.beat_track(audio, trim=False)
    beat_freq = lr.util.fix_frames(beat_freq, x_max=chroma_smooth.shape[1])
    chroma_sync = lr.util.sync(chroma_smooth, beat_freq, aggregate=np.median)
    beat_time = lr.frames_to_time(beat_freq)
    return chroma_sync, beat_time


def get_aggregations(spectrogram, axis=None):
    """Return namedtuple Aggreations with mean, median, mode, max, min, std, sum"""

    Aggregations = namedtuple('Aggregations', 'mean median mode max min std sum')
    aggs = (np.mean, np.median, sp.stats.mode, np.max, np.min, np.std, np.sum)
    return Aggregations(*[agg(spectrogram, axis=axis) for agg in aggs])


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
