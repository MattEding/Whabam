import flask
import numpy as np

import api

from store_segments import retrieve_segments
from utility import find_file


app = flask.Flask(__name__)


@app.route('/')
def index():
    return 'Enter /analyze/<title> to search for a song and display its chroma and tempo graphs'


@app.route('/analyze/<title>')
def analyze(title):
    song = find_file(title)
    npz = retrieve_segments(song)
    segments, name = npz.values()
    name = np.asscalar(name)
    seg0 = segments[0]

    chroma_path = api.save_plot_chroma(seg0, name)
    tempo_path = api.save_plot_tempo(seg0, name)
    audio_path = api.save_segment(seg0, name)

    return flask.render_template('analyze.html',
                                 name=name,
                                 audio_path=audio_path,
                                 chroma_path=chroma_path,
                                 tempo_path=tempo_path)


if __name__ == '__main__':
    app.run(debug=True)
