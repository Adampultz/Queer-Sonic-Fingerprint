import numpy as np
import math
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

def isPowerOf2 (x):
 
    # First x in the below expression 
    # is for the case when x is 0 
    return (x and (not(x & (x - 1))) )
 
def nextPowerOf2(x):
    # Calculate log2 of N
    a = int(math.log2(x))
 
    # If 2^a is equal to N, return N
    if 2**a == x:
        return x
     
    # Return 2^(a + 1)
    return 2**(a + 1)

