#!/usr/bin/python

__author__ = "Matan Lachmish"
__copyright__ = "Copyright 2016, Tel Aviv University"
__version__ = "1.0"
__status__ = "Development"

import numpy as np
import pickle
import sys
import os

import librosa
from librosa import feature

SOUND_SAMPLE_LENGTH = 30000

HAMMING_SIZE = 100
HAMMING_STRIDE = 40
n_fft = 2048  # Length of the FFT window
hop_length = (2 * n_fft) // 3  # Two-thirds of the FFT window length


def die_with_usage():
    """ HELP MENU """
    print('USAGE: python preproccess.py [path to MSD mp3 data]')
    sys.exit(0)


def update_progress(progress):
    print('\r[{0}] {1}%'.format('#' * (progress / 10), progress))


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def prepossessingAudio(audioPath, ppFilePath):
    print('Prepossessing ' + audioPath)

    featuresArray = []
    for i in range(0, SOUND_SAMPLE_LENGTH, HAMMING_STRIDE):
        if i + HAMMING_SIZE <= SOUND_SAMPLE_LENGTH - 1:
            y, sr = librosa.load(audioPath, offset=i / 1000.0, duration=HAMMING_SIZE / 1000.0)


            # Let's make and display a mel-scaled power (energy-squared) spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=n_fft, hop_length=hop_length)

            # Convert to log scale (dB). We'll use the peak power as reference.
            #log_S = librosa.power_to_db(S, ref=np.max)

            #mfcc = librosa.feature.mfcc(S=log_S, sr=sr, n_mfcc=13)
            # featuresArray.append(mfcc)

            featuresArray.append(S)

            if len(featuresArray) == 599:
                break

    print('storing pp file: ' + ppFilePath)

    with open(ppFilePath, 'wb') as f:
        f.write(pickle.dumps(featuresArray))
    f.close()


if __name__ == "__main__":

    # help menu
    if len(sys.argv) < 2:
        die_with_usage()

    i = 0.0
    walk_dir = sys.argv[1]

    print('walk_dir = ' + walk_dir)

    for root, subdirs, files in os.walk(walk_dir):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(root, filename)
                print('\t- file %s (full path: %s)' % (filename, file_path))
                ppFileName = rreplace(file_path, ".wav", ".pp", 1)

                # if os.path.isfile(ppFileName):  # Skip if pp file already exist
                #     continue

                try:
                    prepossessingAudio(file_path, ppFileName)
                except Exception as e:
                    print("Error accured" + str(e))

            if filename.endswith('wav'):
                sys.stdout.write("\r%d%%" % int(1 + i / 1000 * 100))
                sys.stdout.flush()
                i += 1
