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
    """Return dict of chroma_aggs from audio_segment"""

    chroma, _ = utilty.beat_synchronous_chroma(audio_segment)
    chroma_aggs = utilty.get_aggregations(chroma, axis=1)
    chroma_aggs = [utilty.Aggregations(*row) for row in np.array(chroma_aggs).T]
    chroma_ids = 'C Db D Eb E F Gb G Ab A Bb B'.split()
    return {id: agg for id, agg in zip(chroma_ids, chroma_aggs)}
    # return chroma_aggs


def bound_width_aggs(chroma, k=20):
    """Return dict of bound_width_aggs where k is the number of segments to decompose chroma."""

    bound = lr.segment.agglomerative(chroma, k)
    bound_widths = np.diff(bound)
    bound_width_aggs = utilty.get_aggregations(bound_widths, axis=0)
    return {'bound_width': bound_width_aggs}


# TODO: do I want to do Aggreations(tempo) just for consistency?
# Actually just to dict.pop('tempo') before doing stuff with aggs
def tempo_aggs(audio_segment):
    """Return dict of tempogram_aggs, dynamic_tempo_aggs, and tempo from audio_segment"""

    tempogram, dynamic_tempo, tempo= utilty.dynamic_tempo_estimation(audio_segment)
    tempogram_aggs = utilty.get_aggregations(tempogram, axis=1)
    dynamic_tempo_aggs = utilty.get_aggregations(dynamic_tempo)
    tempos = np.asscalar(tempo)
    return {'tempogram': tempogram_aggs, 'dynamic_tempo': dynamic_tempo_aggs, 'tempo': tempo}


def tonnetz_aggs(audio_segment):
    """Return dict of tonnetz_aggs from audio_segment"""

    tonnetz = lr.feature.tonnetz(audio_segment)
    tonnetz_aggs = utilty.get_aggregations(tonnetz, axis=1)
    tonnetz_aggs = [utilty.Aggregations(*row) for row in np.array(tonnetz_aggs).T]
    tonnetz_ids = 'x_P5 y_P5 x_m3 y_m3 x_M3 y_M3'.split()
    return {id: agg for id, agg in zip(tonnetz_ids, tonnetz_aggs)}


def misc_feature_aggs(audio_segment):
    """Return dict of all features from lr.feature with shape (1, t)"""
    
    feature_funcs = [lr.feature.rmse, lr.feature.spectral_bandwidth,
                     lr.feature.spectral_centroid, lr.feature.spectral_flatness,
                     lr.feature.spectral_rolloff, lr.feature.zero_crossing_rate]

    return {feat.__name__: utilty.get_aggregations(feat(audio_segment))
            for feat in feature_funcs}
