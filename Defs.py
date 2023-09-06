import numpy as np
import librosa

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


