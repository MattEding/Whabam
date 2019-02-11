from pathlib import Path

import numpy as np

import utility


cwd = Path.cwd()
data = cwd / 'data'
seg_npz = data / 'seg_npz'


def retrieve_segments(track):
    """Return list of audio_segs for a given track. Pickles them as npz if not already stored."""

    track_npz = seg_npz / (track.stem + '.npz')
    if not track_npz.exists():
        track_segs = utility.load_audio_segments(track)
        track_npz.touch()
        np.savez(track_npz, *track_segs)

    return np.load(track_npz)


if __name__ == '__main__':
    tracks = utility.get_audio_tracks()
    length = len(tracks)
    for i, track in enumerate(tracks[:5]):
        audio_segs = retrieve_segments(track)
        print(f'{i} out of {length} -- file: {track.name}')
