"""
Dataset tools required to reproduce the results of the paper
"Non-Standard  Echo State Networks for Video Door State Monitoring".
"""
import os
import warnings
from pathlib import Path
import numpy as np


class VideoDoorStateRecognitionDataset(object):
    """
    Class to handle the Video Door State Recognition Dataset.

    Parameters
    ----------
    path : str
        /path/to/onset/data
        The path to the video dataset.
    video_suffix : str, default = ".yuv".
        The suffix of the video files.
    annotation_suffix : str, default = ".txt".
        The suffix of the annotation files.
    """
    def __init__(self, path, video_suffix='.yuv', annotation_suffix='.txt'):
        """Construct the VideoDoorStateRecognitionDataset."""
        self.path = path
        # populate lists containing video and annotation files
        video_files = sorted(Path(self.path).rglob(f"*{video_suffix}"))
        annotation_files = sorted(
            Path(self.path).rglob(f"*{annotation_suffix}"))

        self.files = []
        self.video_files = []
        self.annotation_files = []
        for video_file, annotation_file in zip(video_files, annotation_files):
            base_video_file = os.path.splitext(os.path.basename(video_file))
            base_annotation_file = os.path.splitext(
                os.path.basename(annotation_file))
            # search matching audio file
            if base_video_file[0] == base_annotation_file[0]:
                self.video_files.append(video_file)
                self.annotation_files.append(annotation_file)
                # save the base name
                self.files.append(os.path.basename(base_annotation_file[0]))
            else:
                warnings.warn(
                    f"skipping {annotation_file}, no video file found")

    def return_X_y(self):
        """
        Return the dataset in a PyRCN-conform way.

        Returns
        -------
        X : np.ndarray(shape=(n_sequences, ), dtype=object)
            The extracted sequences. Each element of X is a numpy array of
            shape (n_samples, n_features), where n_samples is the sequence
            length.
        y : np.ndarray(shape=(n_sequences, ), dtype=object)
            The pre-processed targets. Each element of y is a numpy array of
            shape (n_samples, 1), where n_samples is the sequence  length.
        """
        X_total = [None] * len(self.video_files)
        y_total = [None] * len(self.video_files)
        for k, (video_file, annotation_file) in enumerate(
                zip(self.video_files, self.annotation_files)):
            X_total[k], y_total[k] = self._pre_process(
                video_file, annotation_file)
        n_sequences_total = [int(len(X) / 5400) for X in X_total]
        X = np.empty(shape=(sum(n_sequences_total), ), dtype=object)
        y = np.empty(shape=(sum(n_sequences_total), ), dtype=object)
        idx = 0
        for X_file, y_file, n_sequences in zip(
                X_total, y_total, n_sequences_total):
            X_split = np.array_split(X_file, n_sequences)
            y_split = np.array_split(y_file, n_sequences)
            for X_seq, y_seq in zip(X_split, y_split):
                X[idx] = X_seq
                y[idx] = y_seq
                idx += 1

        return X, y

    @staticmethod
    def _pre_process(video_file, annotation_file):
        """
        Pre-process the dataset.

        Parameters
        ----------
        video_file : Union[Path, str]
            Full path to the video file to be pre-processed.
        annotation_file : Union[Path, str]
            The onset events, i.e., the labels of the door state.

        Returns
        -------
        X : np.ndarray, shape = (n_samples, n_features)
            The features extracted from the video file.
        y : np.ndarray, shape = (n_samples, )
            The targets that correspond to the video file.
        """
        X = _load_video_file(video_file)
        y = _load_annotations(annotation_file)
        return X, y


def _load_annotations(file_name):
    """
    Load the video annotations from an annotation file.

    Parameters
    ----------
    file_name : Union[Path, str]
        Name of the annotation file to be opened
    """
    file = open(file_name, 'r')
    file_content = list(file.read())
    return np.asarray(file_content, dtype=int)


def _load_video_file(file_name):
    """
    Load the video file.

    Parameters
    ----------
    file_name : Union[Path, str]
        Name of the video file to be opened
    """
    tmp = open(file_name.with_suffix(".txt"), 'rb')
    a = tmp.read()
    tmp.close()
    n_frames = len(a)
    dim = [30, 30]  # Dimension of each frame
    n_features = dim[0] * dim[1]  # size of the input vector
    yuv_file = np.fromfile(file_name, dtype=np.uint8)  # Opening the video file
    X = yuv_file.reshape(n_frames, n_features)
    return X
