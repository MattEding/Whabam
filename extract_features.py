import librosa as lr
import numpy as np

import utilty


def load_audio_segments(file, sec=30):
    """Return audio_segments list and track name from file of duration sec"""

    duration = utilty.get_duration(file)
    segments = int(duration) // sec
    audio_segments = [lr.load(file, offset=sec*i, duration=sec) for i in range(segments)]
    name = file.stem.split('-')[-1].replace('_', ' ').title()
    return audio_segments, name


def chroma_aggs(audio_segment):
    """Return chroma_aggs from audio_segment"""

    chroma, _ = utilty.beat_synchronous_chroma(audio)
    chroma_aggs = utilty.get_aggregations(chroma, axis=1)
    return chroma_aggs


def bound_width_aggs(chroma, k=20):
    """Return bound_width_aggs where k is the number of segments to decompose chroma."""

    bound = lr.segment.agglomerative(chroma, k)
    bound_widths = np.diff(bound)
    bound_width_aggs = utilty.get_aggregations(bound_widths, axis=0)
    return bound_width_aggs


def tempo_aggs(audio_segment):
    """Return tempogram_aggs, dynamic_tempo_aggs, and tempos from audio_segment"""

    tempogram, dynamic_tempo, tempo= utilty.dynamic_tempo_estimation(audio)
    tempogram_aggs = utilty.get_aggregations(tempogram, axis=1)
    dynamic_tempo_aggs = utilty.get_aggregations(dynamic_tempo)
    tempos = np.asscalar(tempo)
    return tempogram_aggs, dynamic_tempo_aggs, tempo


def tonnetz_aggs(audio_segment):
    """Return tonnetz_aggs from audio_segment"""
    tonnetz = lr.feature.tonnetz(audio)
    tonnetz_aggs = utilty.get_aggregations(tonnetz, axis=1)
    tonnetz_map = {0: 'x_P5', 1: 'y_P5', 2: 'x_m3', 3: 'y_m3', 4: 'x_M3', 5: 'y_M3'}
    tonnetz_ids = 'x_P5 y_P5 x_m3 y_m3 x_M3 y_M3'.split()
    return {id: agg for id, agg in zip(tonnetz_ids, tonnetz_aggs._asdict().values())}
    x_P5 = []
    y_P5 = []
    x_m3 = []
    y_m3 = []
    x_M3 = []
    y_M3 = []
    for field, agg in tonnetz_aggs._asdict().items():
