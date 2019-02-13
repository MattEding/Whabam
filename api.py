from pathlib import Path
from string import punctuation

import librosa as lr
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import flask

from utility import beat_synchronous_chroma, dynamic_tempo_estimation


def _format_name(name):
    name = "".join(char for char in name.lower() if char not in punctuation)
    name = name.replace(' ', '_')
    return name


def save_plot_chroma(audio, name):
    """Save beat-synchronous chroma plot to ./static/images and return path"""

    path_name = _format_name(name)
    path = f'static/images/chroma_{path_name}.png'

    if not Path(path).exists():
        #: Need figure, otherwise with flask matplotlib will save on top of old plots
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        chroma_sync, beat_time = beat_synchronous_chroma(audio)
        lr.display.specshow(chroma_sync, x_coords=beat_time, x_axis='time', y_axis='chroma', ax=ax)

        ax.set_title(f'{name}')
        fig.savefig(path, transparent=True, bbox_inches='tight')

    path = f'images/chroma_{path_name}.png'
    return flask.url_for('static', filename=path)


def save_plot_tempo(audio, name):
    """Save tempograph with dynamic tempo plot to ./static/images and return path"""

    path_name = _format_name(name)
    path = f'./static/images/tempo_{path_name}.png'

    if not Path(path).exists():
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        tempogram, dyn_tempo, tempo = dynamic_tempo_estimation(audio)
        lr.display.specshow(tempogram, x_axis='time', y_axis='tempo', ax=ax)

        times = lr.frames_to_time(np.arange(len(dyn_tempo)))
        ax.plot(times, dyn_tempo, 'w')
        ax.plot(times, [tempo] * len(times), '--')

        ax.set_title(f'{name}')
        fig.savefig(path, transparent=True, bbox_inches='tight')

    path = f'images/tempo_{path_name}.png'
    return flask.url_for('static', filename=path)


def save_segment(audio, name):
    """Save wave audio segment to ./static/waves and return path"""

    path_name = _format_name(name)
    path = f'./static/waves/segment_{path_name}.wav'

    if not Path(path).exists():
        lr.output.write_wav(path, audio, sr=22050)

    #: Oddly this will not play in Firefox. Works in Safari and Chrome
    path = f'waves/segment_{path_name}.wav'
    return flask.url_for('static', filename=path)
