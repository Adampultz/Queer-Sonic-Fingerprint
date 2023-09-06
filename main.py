from typing import List, Any

import numpy

import Classes
import Defs
import Defs as defs
import numpy as np
import matplotlib.pyplot as plt
import colorednoise as cn
import scipy.io.wavfile
from numpy.random import default_rng
from datetime import datetime
import scipy.io.wavfile
import os
import librosa
from pythonosc import udp_client

folder = '/March_6th/' # Name of folder for audio files
sendOsc = False  # Send OSC messages to SuperCollider (TRUE/FALSE)
writeFiles = False # Write audio files (TRUE/FALSE)

sampleRate = 48000

hiPassFreq = 20

carrAudioLengthSec = 10
carrAudioLengthSamp = 10 * sampleRate

freqCoeff = carrAudioLengthSamp / sampleRate

brickWallHiPass = int(hiPassFreq * freqCoeff)

client = udp_client.SimpleUDPClient("127.0.0.1", 57120) # For OSC communication

beta = 1

rng = default_rng()

now = datetime.now() # date and time
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") # Date and time as string
dir_path = os.path.dirname(os.path.realpath(__file__)) # Path of project (corresponds to where main.py is located)

# Paths

impulse_output_File_Name = dir_path + folder + dt_string + '_sineKanteen_' + '.wav' # Path of sine wave recording
# object_output_File_Name = dir_path + folder + dt_string + '_kanteen_' + '.wav' # Not in use?
room_output_File_Name = dir_path + folder + dt_string +  '_kanteenImpulseResponse_' + '.wav'
convolution_output_File_Name = dir_path + folder + dt_string + '_kanteenConvolution_ ' + '.wav'
object_audio = dir_path + '/TestAudio/kanteenObject_test.wav'
# carrier_path = dir_path + '/TestAudio/StreetRec_short.wav'
carrier_path = dir_path + '/TestAudio/HofDripMono.wav'
eleStreetNoiseCovolve_audio = dir_path + '/TestAudio/eleStreetNoiseCovolve_audio.wav'
kanteenStreetNoiseCovolve_audio = dir_path + '/TestAudio/kanteenStreetNoiseCovolve_audio.wav'
eleKanteen_audio = dir_path + folder + dt_string + '_elekanteen_ ' + '.wav'
kanteenEle_audio = dir_path + folder + dt_string + '_kanteenEle_ ' + '.wav'
eleKanteenStreetConvolve_audio = dir_path + folder + dt_string + '_elekanteenStreetConcvolve_ ' + '.wav'
kanteenEleStreetConvolve_audio = dir_path + folder + dt_string + '_kanteenEleStreetConvolve_ ' + '.wav'
impulse_path = dir_path + '/TestAudio/sineTest.wav'
obj_path = dir_path + '/Object_Audio/'

numObjects = 0

for subdir, dirs, files in os.walk(obj_path):
    for file in files:
        numObjects += 1

impulse_data, rate = librosa.load(impulse_path, sr = sampleRate)
impulse_data = np.asarray(impulse_data, dtype=np.float64)
impulse_Length = len(impulse_data)

impulse_fftClass = Classes.qsfFFT(impulse_path, sampleRate, impulse_Length, carrAudioLengthSamp)
impulse_rfft = impulse_fftClass.rfft(0)
impulse_fft_Plot = impulse_fftClass.fftPlotNorm()
impulse_rfftSize = impulse_fftClass.size_rfft()

# Arrays

obj_paths_array = []
audio_obj_array = []
obj_1_fft_Plot = []
obj_rfft = np.zeros((numObjects, impulse_rfftSize), dtype=np.complex128)
obj_Response = []
obj_ResponseClass = []

print(carrAudioLengthSamp)



# Iterate over folder with audio recordings of object impulse responses
for subdir, dirs, files in os.walk(obj_path):
    index = 0
    for file in files:
        path = os.path.join(subdir, file) # local

        obj_paths_array.append(path) # local

        obj_FftClass = Classes.qsfFFT(path, sampleRate, impulse_rfftSize, carrAudioLengthSamp) # local
        obj_rfft[index] = obj_FftClass.rfft(brickWallHiPass)

        # obj_Fft.append(filter) # Global
        # Not used
        # audio_obj, rate_obj = librosa.load(path, sr=sampleRate)
        # audio_obj = np.asarray(audio_obj, dtype=np.float64)
        # audio_obj_array.append(audio_obj)

        response = Classes.qsfObjFreqR(obj_rfft[index], impulse_rfft) # Local
        obj_ResponseClass.append(response)
        obj_Response.append(response.response())

        index += 1

responseLength = len(obj_Response[0])

print(obj_rfft[300:320])


objChild = np.zeros((2,2, responseLength), dtype=np.complex128)
sourceOBJ_convo = np.zeros((2, responseLength), dtype=np.complex128)
newGen = np.zeros((2, responseLength), dtype=np.complex128)
newGen_CarrierConvo = np.zeros((2, responseLength), dtype=np.complex128)

audioIndex = 0
audioIndex_Object = 0

# Source object convolution with carrier audio





# sine_data, rate = librosa.load(sine_audio, sr = sampleRate)
# sine_data = np.asarray(sine_data, dtype=np.float64)
#
# sineFftClass = Classes.qsfFFT(sine_audio, sampleRate, fftLength, carrAudioLengthSamp)
# sineFft = sineFftClass.rfft(0)
# sine_fft_Plot = sineFftClass.fftPlotNorm()

# for i in range(numObjects):
#     audio_obj, rate_obj = librosa.load(obj_paths_array[i], sr = sampleRate)
#     audio_obj = np.asarray(audio_obj, dtype = np.float64)
#     audio_obj_array.append(audio_obj)
#
#     obj_Response.append(Classes.qsfObjFreqR(obj_Fft[i], sineFft))
    # kanteenResponse = Classes.qsfObjFreqR(obj_Fft[0], sineFft)

    # obj_1_fft_Plot = obj_1_FftClass.fftPlotNorm()



# audio_obj_0, rate_obj_0 = librosa.load(obj_paths_array[0], sr = sampleRate)
# audio_obj_0 = np.asarray(audio_obj_0, dtype = np.float64)
# audio_obj_1, rate_obj_1 = librosa.load(obj_paths_array[1], sr = sampleRate)
# audio_obj_1 = np.asarray(audio_obj_1, dtype = np.float64)

audio_street, streetsRate = librosa.load(carrier_path, sr = sampleRate)
audio_street = np.asarray(audio_street, dtype = np.float64)

impulse_WinSize = len(impulse_data)
# objectWinSize = len(aud_data_Object)

audio_street = audio_street[0:impulse_WinSize]

slicePoint = int(1000 * freqCoeff)

ii = np.arange(0, impulse_WinSize)
t = ii / sampleRate

len_data = len(impulse_path)

fftLength = 168000

# obj_1_FftClass = Classes.qsfFFT(obj_paths_array[1], sampleRate, fftLength, carrAudioLengthSamp)
# obj_1_Fft = obj_1_FftClass.rfft(brickWallHiPass)
# obj_1_fft_Plot = obj_1_FftClass.fftPlotNorm()

# obj_0_FftClass = Classes.qsfFFT(obj_paths_array[0], sampleRate, fftLength, carrAudioLengthSamp)
# obj_0_Fft = obj_0_FftClass.rfft(brickWallHiPass)
# obj_0_fft_Plot = obj_0_FftClass.fftPlotNorm()

carrier_FftClass = Classes.qsfFFT(carrier_path, sampleRate, carrAudioLengthSamp, carrAudioLengthSamp)
carrier_Fft = carrier_FftClass.rfft(brickWallHiPass)
carrier_fft_Plot = carrier_FftClass.fftPlotNorm()

for i in range(numObjects):
    sourceOBJ_convo[i] = (obj_Response[i] * carrier_Fft).copy()

# Genetic Operators

for i in range(numObjects):
    child_1 = obj_Response[i][0:slicePoint].copy()
    child_2 = obj_Response[(i + 1) % 2][(slicePoint):len(obj_rfft[1])].copy()
    newGen[i] = np.concatenate((child_1, child_2))

for i in range(numObjects):
    newGen_CarrierConvo[i] = (newGen[i] * carrier_Fft)
# eleKanteenConcatStreetConvolve = eleKanteenConcat * carrier_Fft
# kanteenEleConcatStreetConvolve = kanteenEleConcat * carrier_Fft


sine_ifft = np.fft.irfft(impulse_rfft)
sine_ifft = sine_ifft[0:impulse_WinSize]
sine_ifft = np.real(sine_ifft)

obj_1_ifft = np.fft.irfft(obj_rfft[1])
obj_1_ifft = obj_1_ifft[0:impulse_WinSize]
obj_1_ifft = np.real(obj_1_ifft)
elephResponse_ifft = np.fft.irfft(obj_Response[1])
elephResponse_ifft = elephResponse_ifft[0:impulse_WinSize]
elephResponse_ifft = np.real(elephResponse_ifft)
obj_0_ifft = np.fft.irfft(obj_rfft[0])
obj_0_ifft = obj_0_ifft[0:impulse_WinSize]
obj_0_ifft = np.real(obj_0_ifft)
kanteenResponse_ifft = np.fft.irfft(obj_Response[0])
kanteenResponse_ifft = kanteenResponse_ifft[0:impulse_WinSize]
kanteenResponse_ifft = np.real(kanteenResponse_ifft)
elekanteenResponse_ifft = np.fft.irfft(newGen[1])
elekanteenResponse_ifft = elekanteenResponse_ifft[0:impulse_WinSize]
elekanteenResponse_ifft = np.real(elekanteenResponse_ifft)
kanteenEleResponse_ifft = np.fft.irfft(newGen[0])
kanteenEleResponse_ifft = kanteenEleResponse_ifft[0:impulse_WinSize]
kanteenEleResponse_ifft = np.real(kanteenEleResponse_ifft)
streeNoise_ifft = np.fft.irfft(carrier_Fft)
streeNoise_ifft = np.real(streeNoise_ifft)
elphStreetConvolveIrfft = np.fft.irfft(sourceOBJ_convo[1])
kanteenStreetConvolveIrfft = np.fft.irfft(sourceOBJ_convo[0])
eleKanteenConcatStreetConvolveIrfft = np.fft.irfft(newGen_CarrierConvo[1])
kanteenEleConcatStreetConvolveIrfft = np.fft.irfft(newGen_CarrierConvo[0])

w = np.linspace(0, sampleRate // 2, impulse_rfftSize)

audioPltLn = np.linspace(0, impulse_WinSize // sampleRate, impulse_WinSize)

carrierPltLn = np.linspace(0, carrAudioLengthSamp // sampleRate, carrAudioLengthSamp)

# fig, axs = plt.subplots(3, 2, figsize = (20, 10))
#
# # axs[0].plot(audioPltLn, sineFftClass.audio())
# # axs[0].set_ylabel('Amplitude')
# # axs[0].title.set_text('Sine wave sweep (time domain)')
# #
# # axs[1].plot(audioPltLn, elephantFftClass.audio())
# # axs[1].set_ylabel('Amplitude')
# # axs[1].title.set_text('Elephant impulse response (time domain)')
# #
# # axs[2].plot(audioPltLn, kanteenFftClass.audio())
# # axs[2].set_xlabel('Time in seconds')
# # axs[2].set_ylabel('Amplitude')
# # axs[2].title.set_text('Thermos impulse response (time domain)')
#
# axis_1 = axs[0, 0]
# axis_1.plot(audioPltLn, sineFftClass.audio())
# axis_1.set_ylabel('Amplitude')
# axis_1.title.set_text('Sine wave sweep (time domain)')
#
# axis_1R = axs[0, 1]
# axis_1R.plot(w, sine_fft_Plot)
# axis_1R.title.set_text('Sine wave sweep (frequency domain)')
#
# axis_2 = axs[1, 0]
# axis_2.plot(audioPltLn, elephantFftClass.audio())
# axis_2.set_ylabel('Amplitude')
# axis_2.title.set_text('Elephant impulse response (time domain)')
#
# axis_2R = axs[1, 1]
# axis_2R.plot(w, elephant_fft_Plot)
# axis_2R.title.set_text('Elephant frequency response (frequency domain)')
#
# axis_3 = axs[2, 0]
# axis_3.plot(audioPltLn, kanteenFftClass.audio())
# axis_3.set_ylabel('Amplitude')
# axis_3.title.set_text('Thermos impulse response (time domain)')
#
# axis_3R = axs[2, 1]
# axis_3R.plot(w, kanteen_fft_Plot)
# axis_3R.title.set_text('Thermos frequency response (frequency domain)')
# axis_3R.set_xlabel('Frequency in Hz')
#
# fig, axs = plt.subplots(2, 3, figsize = (20, 10))
#
# axis_1 = axs[0, 0]
# line_1 = axis_1.plot(w, sine_fft_Plot, label='Sine Sweep')
#
# line_2 = axis_1.plot(w, elephant_fft_Plot, label='Elephant')
# axis_1.legend()
# axis_1.set_ylabel('Amplitude')
# axis_1.title.set_text('Sine sweep and elephant (frequency domain)')
#
# axis_1_2 = axs[0, 1]
# axis_1_2.plot(w, elphResponse.absNorm())
# axis_1_2.title.set_text('Elephant sonic fingerprint (frequency domain)')
#
# axis_1_3 = axs[0, 2]
# axis_1_3.plot(audioPltLn, elephResponse_ifft)
# axis_1_3.title.set_text('Elephant sonic fingerprint (time domain)')
#
# axis_2 = axs[1, 0]
# axis_2.set_ylabel('Amplitude')
# line_1 = axis_2.plot(w, sine_fft_Plot, label='Sine Sweep')
#
# line_2 = axis_2.plot(w, kanteen_fft_Plot, label='Thermos')
# axis_2.legend()
# axis_2.title.set_text('Sine sweep and thermos (frequency domain)')
# axis_2.set_xlabel('Frequency in Hz')
#
# axis_2_2 = axs[1, 1]
# axis_2_2.plot(w, kanteenResponse.absNorm())
# axis_2_2.title.set_text('Thermos sonic fingerprint (frequency domain)')
# axis_2_2.set_xlabel('Frequency in Hz')
#
# axis_2_3 = axs[1, 2]
# axis_2_3.plot(audioPltLn, librosa.util.normalize(kanteenResponse_ifft))
# axis_2_3.title.set_text('Thermos sonic fingerprint (time domain)')
# axis_2_3.set_xlabel('Time in seconds')

#
# axis_2R = axs[1, 1]
# axis_2R.plot(w, elephant_fft_Plot)
# axis_2R.title.set_text('Elephant frequency response (frequency domain)')
#
# axis_3 = axs[2, 0]
# axis_3.plot(audioPltLn, kanteenFftClass.audio())
# axis_3.set_ylabel('Amplitude')
# axis_3.title.set_text('Thermos impulse response (time domain)')
#
# axis_3R = axs[2, 1]
# axis_3R.plot(w, kanteen_fft_Plot)
# axis_3R.title.set_text('Thermos frequency response (frequency domain)')
# axis_3R.set_xlabel('Frequency in Hz')

# axs[1].plot(audioPltLn, elephantFftClass.audio())
# axs[1].set_ylabel('Amplitude')
# axs[1].title.set_text('Elephant impulse response (time domain)')
#
# axs[2].plot(audioPltLn, kanteenFftClass.audio())
# axs[2].set_xlabel('Time in seconds')
# axs[2].set_ylabel('Amplitude')
# axs[2].title.set_text('Thermos impulse response (time domain)')

# fig, axs = plt.subplots(5, 3, figsize = (20, 10))
#
# axs[0, 0].plot(audioPltLn, sineFftClass.audio())
# axs[0, 1].plot(w, sine_fft_Plot)
# axs[0, 2].plot(audioPltLn, sine_ifft)
#
# axs[1, 0].plot(audioPltLn, elephantFftClass.audio())
# axs[1, 1].plot(w, elephant_fft_Plot)
# axs[1, 2].plot(audioPltLn, eleph_ifft)
#
# axs[2, 0].plot(audioPltLn, kanteenFftClass.audio())
# axs[2, 1].plot(w, kanteen_fft_Plot)
# axs[2, 2].plot(audioPltLn, kanteen_ifft)
#
# axs[3, 0].plot(carrierPltLn, streetNoiseFftClass.audio())
# axs[3, 1].plot(w, streetNoise_fft_Plot)
# axs[3, 2].plot(carrierPltLn, streeNoise_ifft)

# fig, axs = plt.subplots(2, 2, figsize = (20, 10))
#
# axis_1 = axs[0, 0]
# axis_1.plot(carrierPltLn, streetNoiseFftClass.audio())
# axis_1.title.set_text('Snow thawing (time domain)')
# axis_1.set_ylabel('Amplitude')
# axis_1.set_xlabel('Time in seconds')
#
# axis_1_2 = axs[0, 1]
# axis_1_2.plot(w, streetNoiseFftClass.fftPlotNorm())
# axis_1_2.title.set_text('Snow thawing (frequency domain)')
# axis_1_2.set_xlabel('Frequency in Hz')
#
# axis_1_1 = axs[0, 0]
# axis_1_1.plot(w, streetNoiseFftClass.fftPlotNorm(), label = 'Snow thawing')
# axis_1_1.plot(w, elephant_fft_Plot, label = 'Elephant fingerprint')
# axis_1_1.title.set_text('Snow thawing x elephant (frequency domain)')
# axis_1_1.legend()
#
# axis_1_2 = axs[0, 1]
# axis_1_2.plot(carrierPltLn, librosa.util.normalize(elphStreetConvolveIrfft), label = 'Audio with elephant fingerprint')
# axis_1_2.plot(carrierPltLn, streetNoiseFftClass.audio(), label = 'Original audio')
# axis_1_2.title.set_text('Snow thawing x elephant (time domain)')
# axis_1_2.legend()
#
# axis_2_1 = axs[1, 0]
# axis_2_1.plot(w, streetNoiseFftClass.fftPlotNorm(), label = 'Snow thawing')
# axis_2_1.plot(w, kanteen_fft_Plot, label = 'Thermos fingerprint')
# axis_2_1.title.set_text('Snow thawing x thermos (frequency domain)')
# axis_2_1.set_xlabel('Frequency in Hz')
# axis_2_1.legend()
#
# axis_2_2 = axs[1, 1]
# axis_2_2.plot(carrierPltLn, librosa.util.normalize(kanteenStreetConvolveIrfft), label = 'Audio with thermos fingerprint')
# axis_2_2.plot(carrierPltLn, streetNoiseFftClass.audio(), label = 'Original audio')
# axis_2_2.title.set_text('Snow thawing x thermos (time domain)')
# axis_2_2.set_xlabel('Time in seconds')
# axis_2_2.legend()
#
# axis_2 = axs[1, 0]
# axis_2.set_ylabel('Amplitude')
# line_1 = axis_2.plot(w, sine_fft_Plot, label='Sine Sweep')
#
# line_2 = axis_2.plot(w, kanteen_fft_Plot, label='Thermos')
# axis_2.legend()
# axis_2.title.set_text('Sine sweep and thermos (frequency domain)')
# axis_2.set_xlabel('Frequency in Hz')
#
# axis_2_2 = axs[1, 1]
# axis_2_2.plot(w, kanteenResponse.absNorm())
# axis_2_2.title.set_text('Thermos sonic fingerprint (frequency domain)')
# axis_2_2.set_xlabel('Frequency in Hz')
#
# axis_2_3 = axs[1, 2]
# axis_2_3.plot(audioPltLn, kanteen_ifft)
# axis_2_3.title.set_text('Thermos sonic fingerprint (time domain)')
# axis_2_3.set_xlabel('Time in seconds')

fig, axs = plt.subplots(2, 3, figsize = (20, 10))

axis_1_2 = axs[0, 0]
axis_1_2.plot(w, obj_ResponseClass[1].absNorm())
axis_1_2.vlines(x = 1500, ymin = 0, ymax = 0.5,
           colors = 'red',
           label = 'slice point')
axis_1_2.title.set_text('Elephant sonic fingerprint (frequency domain)')
axis_1_2.legend()

axis_1_2 = axs[1, 0]
axis_1_2.plot(w, obj_ResponseClass[0].absNorm())
axis_1_2.vlines(x = 1500, ymin = 0, ymax = 0.5,
           colors = 'red',
           label = 'slice point')
axis_1_2.title.set_text('Thermos sonic fingerprint (frequency domain)')
axis_1_2.set_xlabel('Frequency in Hz')
axis_1_2.legend()

axis_1_2 = axs[0, 1]
# axis_1_2.plot(w, librosa.util.normalize(abs(eleKanteenConcat)))
axis_1_2.plot(w, librosa.util.normalize(np.concatenate((obj_ResponseClass[1].absNorm()[0:slicePoint], obj_ResponseClass[0].absNorm()[slicePoint:len(obj_rfft[0])]))))
axis_1_2.title.set_text('Elephant / Thermos sonic fingerprint (frequency domain)')

axis_1_2 = axs[1, 1]
axis_1_2.plot(w, np.concatenate((obj_ResponseClass[0].absNorm()[0:slicePoint], obj_ResponseClass[1].absNorm()[slicePoint:len(obj_rfft[0])])))
axis_1_2.title.set_text('Thermos / Elephant sonic fingerprint (frequency domain)')
axis_1_2.set_xlabel('Frequency in Hz')

axis_1_2 = axs[0, 2]
axis_1_2.plot(audioPltLn, librosa.util.normalize(elekanteenResponse_ifft))
axis_1_2.title.set_text('Elephant / Thermos sonic fingerprint (time domain)')

axis_1_2 = axs[1, 2]
axis_1_2.plot(audioPltLn, librosa.util.normalize(kanteenEleResponse_ifft))
axis_1_2.title.set_text('Thermos / Elephant sonic fingerprint (time domain)')
axis_1_2.set_xlabel('Time in seconds')

# axs[1,0].plot(audioPltLn, channel_2)
# axs[1, 1].plot(w, fourierObjectAbs)

# fig, axs = plt.subplots(2, 2, figsize = (20, 10))
#
# axis_2_2 = axs[0, 0]
# axis_2_2.plot(carrierPltLn, librosa.util.normalize(eleKanteenConcatStreetConvolveIrfft), label = 'Audio with elephant / thermos fingerprint')
# axis_2_2.plot(carrierPltLn, streetNoiseFftClass.audio(), label = 'Original audio')
# axis_2_2.title.set_text('Snow thawing x elephant / thermos (time domain)')
# axis_2_2.set_xlabel('Time in seconds')
# axis_2_2.legend()
#
# axis_2_2 = axs[1, 0]
# axis_2_2.plot(carrierPltLn, librosa.util.normalize(kanteenEleConcatStreetConvolveIrfft), label = 'Audio with thermos / elephant fingerprint')
# axis_2_2.plot(carrierPltLn, streetNoiseFftClass.audio(), label = 'Original audio')
# axis_2_2.title.set_text('Snow thawing x thermos / elephant (time domain)')
# axis_2_2.set_xlabel('Time in seconds')
# axis_2_2.legend()

#
# for i in range(int(20 * freqCoeff)):
#     convolve[i] = 0

# convolve[int(2000 * freqCoeff)] = -1-0.06659464496026408j

# print(convolve[int(2000 * freqCoeff)])

# resFreqsPlot = convolve_abs * pitches
#

# newGen_1NoisConVolve = elphResponse * streetNoiseFft
# newGen_1NoisConVolve = librosa.util.normalize(newGen_1NoisConVolve)



# axs[2, 1].plot(w, convolve_abs)

# division = fourierObject / fourier
#
# roomImpulse = np.fft.irfft(division)
# roomImpulse = np.real(roomImpulse)
# roomImpulse = librosa.util.normalize(roomImpulse)


# fig, axs = plt.subplots(5, 3, figsize = (20, 10))
#
# # axs[0, 0].plot(audioPltLn, )
#
# axs[0, 0].plot(w, elphResponse.absNorm())
# axs[0, 1].plot(w, abs(elphStreetConvolve))
#
# axs[1, 0].plot(w, kanteenResponse.absNorm())
# axs[1, 1].plot(w, abs(kanteenStreetConvolve))

# axs[2, 1].plot(w, librosa.util.normalize(abs(newGen_1)))
#
# axs[3, 0].plot(w, kanteen_fft_Plot)
#
# axs[3, 1].plot(w, librosa.util.normalize(abs(newGen_1NoisConVolve)))
#
# newGenAduio = Defs.qsfWriteIfft(newGen_1NoisConVolve)
#
# newGen_1_ifft = np.fft.irfft(newGenAduio)
#
# newGen_2_ifft = np.fft.irfft(newGen_2)

plt.show()

#
# scipy.io.wavfile.write(kanteenEle, sampleRate, librosa.util.normalize(np.real(newGen_2_ifft)))


# if (writeFiles == True):
#
#     # scipy.io.wavfile.write(eleStreetNoiseCovolve_audio, sampleRate, librosa.util.normalize(np.real(elphStreetConvolveIrfft)))
#     # scipy.io.wavfile.write(kanteenStreetNoiseCovolve_audio, sampleRate,
#     #                        librosa.util.normalize(np.real(kanteenStreetConvolveIrfft)))
#     # scipy.io.wavfile.write(eleKanteenStreetConvolve_audio, sampleRate,
#     #                        librosa.util.normalize(np.real(eleKanteenConcatStreetConvolveIrfft)))
#     # scipy.io.wavfile.write(kanteenEleStreetConvolve_audio, sampleRate,
#     #                        librosa.util.normalize(np.real(kanteenEleConcatStreetConvolveIrfft)))
#     # scipy.io.wavfile.write(eleKanteen_audio, sampleRate, librosa.util.normalize(elekanteenResponse_ifft))
#     # scipy.io.wavfile.write(kanteenEle_audio, sampleRate, librosa.util.normalize(kanteenEleResponse_ifft))

