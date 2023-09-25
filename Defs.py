import numpy as np
import librosa
from pynput import keyboard


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def qsfWriteIfft(arr):
    irfft = np.fft.irfft(arr)
    real = np.real(irfft)
    norm = librosa.util.normalize(real)
    return norm

def scale_number(unscaled, from_min, from_max, to_min, to_max):
    return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min

def trigNewGen():
    def keyPress(key):
        if key == keyboard.Key.space:
            return 1

    def on_release(key):
        print('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            return 0

    listener = keyboard.Listener(on_press=keyPress, on_release=on_release)
    listener.start()
