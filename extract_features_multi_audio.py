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


def chroma_aggs(audio_segments):
    """Return chroma_aggs from audio_segments"""

    chroma_syncs = [utilty.beat_synchronous_chroma(audio) for audio, _ in audio_segments]
    chromas, _ = zip(*chroma_syncs)
    chroma_aggs = [utilty.get_aggregations(chroma, axis=1) for chroma in chromas]
    return chroma_aggs


def bound_width_aggs(chromas, k=20):
    """Return bound_width_aggs where k is the number of segments to decompose chromas."""

    bounds = [lr.segment.agglomerative(chroma, k) for chroma in chromas]

    # not list comp? fix?
    bound_widths = np.diff(bounds)
    bound_width_aggs = utilty.get_aggregations(bound_widths, axis=0)
    return bound_width_aggs


def tempo_aggs(audio_segments):
    """Return tempogram_aggs, dynamic_tempo_aggs, and tempos from audio_segments"""

    dynamic_tempograms = [utilty.dynamic_tempo_estimation(audio) for audio, _ in audio_segments]
    tempograms, dynamic_tempos, tempos = zip(*dynamic_tempograms)
    tempogram_aggs = [utilty.get_aggregations(tempogram, axis=1) for tempogram in tempograms]
    dynamic_tempo_aggs = [utilty.get_aggregations(dyn_tempo) for dyn_tempo in dynamic_tempos]
    tempos = [np.asscalar(tempo) for tempo in tempos]
    return tempogram_aggs, dynamic_tempo_aggs, tempos


def tonnetz_aggs(audio_segments):
    """Return tonnetz_aggs from audio_segments"""
    tonnetzes = [lr.feature.tonnetz(audio) for audio, _ in audio_segments]
    tonnetz_aggs = [utilty.get_aggregations(tonnetz, axis=1) for tonnetz in tonnetzes]
    tonnetz_map = {0: 'x_P5', 1: 'y_P5', 2: 'x_m3', 3: 'y_m3', 4: 'x_M3', 5: 'y_M3'}
