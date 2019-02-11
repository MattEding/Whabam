from collections import ChainMap

import librosa as lr
import numpy as np

import utility


def chroma_aggs(audio_segment):
    """Return dict of chroma_aggs from audio_segment"""

    chroma, _ = utility.beat_synchronous_chroma(audio_segment)
    chroma_aggs = utility.get_aggregations(chroma, axis=1)
    chroma_aggs = [utility.Aggregations(*row) for row in np.array(chroma_aggs).T]
    # chroma_ids = 'C Db D Eb E F Gb G Ab A Bb B'.split()
    chroma_ids = range(12)
    return {f'pitch_{id}': agg for id, agg in zip(chroma_ids, chroma_aggs)}


def bound_width_aggs(chroma, k=20):
    """Return dict of bound_width_aggs where k is the number of segments to decompose chroma"""

    bound = lr.segment.agglomerative(chroma, k)
    bound_widths = np.diff(bound)
    bound_width_aggs = utility.get_aggregations(bound_widths, axis=0)
    return {'bound_width': bound_width_aggs}


# TODO: do I want to do Aggreations(tempo) just for consistency?
# Actually just to dict.pop('tempo') before doing stuff with aggs
def tempo_aggs(audio_segment):
    """Return dict of tempogram_aggs, dynamic_tempo_aggs, and tempo from audio_segment"""

    tempogram, dynamic_tempo, tempo= utility.dynamic_tempo_estimation(audio_segment)
    tempogram_aggs = utility.get_aggregations(tempogram)#, axis=1
    dynamic_tempo_aggs = utility.get_aggregations(dynamic_tempo)
    tempos = np.asscalar(tempo)
    return {'tempogram': tempogram_aggs, 'dynamic_tempo': dynamic_tempo_aggs, 'tempo': tempo}


def tonnetz_aggs(audio_segment):
    """Return dict of tonnetz_aggs from audio_segment"""

    tonnetz = lr.feature.tonnetz(audio_segment)
    tonnetz_aggs = utility.get_aggregations(tonnetz, axis=1)
    tonnetz_aggs = [utility.Aggregations(*row) for row in np.array(tonnetz_aggs).T]
    tonnetz_ids = 'x_perf5 y_perf5 x_min3 y_min3 x_maj3 y_maj3'.split()
    return {id: agg for id, agg in zip(tonnetz_ids, tonnetz_aggs)}


def misc_feature_aggs(audio_segment):
    """Return dict of all features from lr.feature with shape (1, t) from audio_segment"""

    magnitude = abs(lr.stft(audio_segment))

    feature_funcs_y = [lr.feature.rmse, lr.feature.zero_crossing_rate]
    feature_funcs_S = [lr.feature.spectral_bandwidth, lr.feature.spectral_centroid,
                       lr.feature.spectral_flatness, lr.feature.spectral_rolloff]

    features_y = {feat.__name__: utility.get_aggregations(feat(y=audio_segment))
                  for feat in feature_funcs_y}
    features_S = {feat.__name__: utility.get_aggregations(feat(S=magnitude))
                  for feat in feature_funcs_S}

    return {**features_y, **features_S}


def extract_all(audio_segment):
    """Return ChainMap of all features from audio_segment"""

    chroma_dct = chroma_aggs(audio_segment)
    bound_width_dct = bound_width_aggs(audio_segment)
    tempo_dct = tempo_aggs(audio_segment)
    tonnetz_dct = tonnetz_aggs(audio_segment)
    misc_feature_dct = misc_feature_aggs(audio_segment)
    # tempo_dct in first map to allow pop of tempo key
    return ChainMap(tempo_dct, chroma_dct, bound_width_dct, tonnetz_dct, misc_feature_dct)
