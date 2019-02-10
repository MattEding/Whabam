from pathlib import Path

import numpy as np

import utility


cwd = Path.cwd()
data = cwd / 'data'
seg_npz = data / 'seg_npz'


def retrieve_segments(track):
    """Return list of audio_segs for a given track"""

    track_npz = seg_npz / (track.stem + '.npz')#.replace('-', '_')
    if track_npz.exists():
        # track_segs = np.load(track_npz)
        pass
    else:
        track_segs = utility.load_audio_segments(track)
        track_npz.touch()
        np.savez(track_npz, *track_segs)

    return np.load(track_npz)

    # return track_segs


if __name__ == '__main__':
    # load all track segments into shelve db
    tracks = utility.get_audio_tracks()
    length = len(tracks)
    for i, track in enumerate(tracks[:5]):
        audio_segs = retrieve_segments(track)
        print(f'{i} out of {length} -- file: {track.name}')
