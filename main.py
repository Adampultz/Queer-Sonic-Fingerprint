import numpy
import Classes
import qsf_evolutionaryOperators as evoOp
import numpy as np
from numpy.random import default_rng
from datetime import datetime
import os
import librosa
import scipy
from pythonosc import udp_client
from subprocess import call
from pynput import keyboard
import Plotting as qsfPlot
import asyncio
import math

action = 0

populationLimit = 50

# print(defs.keyPress(keyboard))

# defs.on_release(key)

# def keyPress(key):
#     if key == key.space:
#         print('trigger')

    # def on_release(key):
    #     print('{0} released'.format(
    #         key))
    #     if key == keyboard.Key.esc:
    #         return 0

# listener = keyboard.Listener(on_press=keyPress)
    # listener = keyboard.Listener(on_press=keyPress, on_release=on_release)
# listener.start()


sendOsc = False  # Send OSC messages to SuperCollider (TRUE/FALSE)
writeFiles = True  # Write audio files (TRUE/FALSE)
visualise = True  # Write audio files (TRUE/FALSE)
run_evolution = False # Generate new generations

sampleRate = 48000

nyquist = sampleRate / 2

hiPassFreq = 20

genID = 0

carrAudioLengthSec = 10
carrAudioLengthSamp = carrAudioLengthSec * sampleRate

freqCoef = carrAudioLengthSamp / sampleRate

brickWallHiPass = hiPassFreq

client = udp_client.SimpleUDPClient("127.0.0.1", 57120)  # For OSC communication

# beta = 1

rng = default_rng()

now = datetime.now()  # date and time
date_string = now.strftime("%d_%m_%Y")
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")  # Date and time as string
dir_path = os.path.dirname(os.path.realpath(__file__))  # Path of project (corresponds to where main.py is located)
dir_path = os.path.dirname(dir_path)
dir_path = os.path.dirname(dir_path)
dir_path = os.path.dirname(dir_path)

audioFolder = '/Audio' # Name of folder for audio files
input_folder = audioFolder + '/Input'  
impulse_folder = input_folder + '/Impulses' # Folder for impulses (noise burst and sine sweeps)
impulseResponse_folder = audioFolder + '/Impulse_Responses' # Folder for impulses (noise burst and sine sweeps)
output_folder = audioFolder + '/Output'  # Folder for audio output files
carrierFolder = audioFolder + '/CarrierAudio'

impulse_path = dir_path + impulse_folder
# obj_path = dir_path + '/Object_Audio_NoiseRedux/'
# object_audio = dir_path + '/TestAudio/kanteenObject_test.wav'
carrier_path = dir_path + carrierFolder + '/HofDripMono.wav'

sineSweepExpPath = impulse_path + '/sineTest.wav'
# impulseResponsePath = dir_path + impulseResponse_folder + '/Object_Audio_NoiseRedux'
impulseResponsePath = dir_path + impulseResponse_folder + '/Object_Audio_DI'

ImpulseResponses = Classes.qsf_ImpulseResponses(impulseResponsePath)
numObjects = ImpulseResponses.numbObjects()
objectNames = np.ndarray((numObjects,), dtype=object)
obj_paths_array = []
obj_FftClass = numpy.ndarray((numObjects,), dtype=object)
obj_maxSizeSamp = 0


for subdir, dirs, files in os.walk(impulseResponsePath):
    index = 0
    for file in files:
        if file == '.DS_Store':
            print("Ignoring file '.DS_Store'")
        else:
            objFileName = ''
            path = os.path.join(subdir, file)  # local

            for char in range(len(file)):
                if file[char + 2] == '_':
                    objectNames[index] = objFileName
                    break
                else:
                    objFileName += file[char + 2]

            obj_paths_array.append(path)  # local

            obj_FftClass[index] = Classes.qsfFFT(path, sampleRate, carrAudioLengthSamp, freqCoef)

            obj_sizeSamp = obj_FftClass[index].sizeSamples()

            if obj_sizeSamp > obj_maxSizeSamp:
                obj_maxSizeSamp = obj_sizeSamp
            
            # objFft = obj_FftClass[index].rfft(brickWallHiPass)
            # objFft_size = len(objFft)

            # obj_rfft.append(objFft)
            # obj_fft_Plot.append(obj_FftClass[index].fftPlotNorm())

            # response = Classes.qsfObjFreqR(obj_rfft[index], impulse_rfft)  # Local
            # obj_ResponseClass[index] = response
            # obj_Response[index] = response.response()

            index += 1

impulse = sineSweepExpPath # Used for changing impulse, for example between noise burst, sine sweep, and exponential sine sweep

impulse_data, rate = librosa.load(impulse, sr=sampleRate)
impulse_data_sizeSamp = len(impulse_data)

impulse_fftClass = Classes.qsfFFT(impulse, sampleRate, carrAudioLengthSamp, freqCoef)

if impulse_data_sizeSamp < obj_maxSizeSamp:
    impulse_fftClass.zeroPadAudio(obj_maxSizeSamp)

impulse_rfft = impulse_fftClass.rfft(0)
impulse_rfftSize = impulse_fftClass.size_rfft()
impulse_fft_Plot = impulse_fftClass.fftPlotNorm()

# Arrays

obj_fft_Plot = np.zeros((numObjects, impulse_rfftSize))
obj_rfft = np.zeros((numObjects, impulse_rfftSize), dtype=np.complex128)
newGen_CarrierConvo = np.zeros((numObjects, impulse_rfftSize), dtype=np.complex128)
obj_irfft = np.zeros((numObjects, obj_maxSizeSamp))
newGen_irfft = np.zeros((populationLimit, obj_maxSizeSamp))
impulse_irfft = np.zeros((populationLimit, obj_maxSizeSamp))
obj_carr_convo_irfft = np.zeros((numObjects, obj_maxSizeSamp))
newGen_carr_convo_irfft = np.zeros((populationLimit, obj_maxSizeSamp))


axisArray = numpy.ndarray((4, 4), dtype=object)
obj_Response = np.zeros((numObjects, impulse_rfftSize), dtype=np.complex128)
obj_ResponseClass = numpy.ndarray((numObjects,), dtype=object)

# Iterate over folder with audio recordings of object impulse responses

for subdir, dirs, files in os.walk(impulseResponsePath):
    index = 0
    for file in files:
        if file == '.DS_Store':
            print("Ignoring file '.DS_Store'")
        else:
            objFileName = ''
            path = os.path.join(subdir, file)  # local

            # for char in range(len(file)):
            #     if file[char + 2] == '_':
            #         objectNames[index] = objFileName
            #         break
            #     else:
            #         objFileName += file[char + 2]

            # obj_paths_array.append(path)  # local

            obj_FftClass[index] = Classes.qsfFFT(path, sampleRate, carrAudioLengthSamp, freqCoef)

            if obj_FftClass[index].sizeSamples() < obj_maxSizeSamp:
                obj_FftClass[index].zeroPadAudio(obj_maxSizeSamp)

            obj_rfft[index] = obj_FftClass[index].rfft(brickWallHiPass)
            obj_fft_Plot[index] = obj_FftClass[index].fftPlotNorm()
            response = Classes.qsfObjFreqR(obj_rfft[index], impulse_rfft)  # Local
            obj_ResponseClass[index] = response
            obj_Response[index] = response.response()

            index += 1

responseLength = len(obj_Response[0])

objChild = np.zeros((numObjects, numObjects, responseLength), dtype=np.complex128)
newGen = np.zeros((numObjects, responseLength), dtype=np.complex128)
newGenNoMutation = np.zeros((numObjects, responseLength), dtype=np.complex128)

audioIndex = 0
audioIndex_Object = 0

# Source object convolution with carrier audio

audio_carrier, carrier_sRate = librosa.load(carrier_path, sr=sampleRate)
audio_carrier = np.asarray(audio_carrier, dtype=np.float64)

audio_carrier = audio_carrier[0:obj_maxSizeSamp]

# len_data = len(len_data)

# print(len(obj_Response[0]))
# print(len(audio_carrier))

# fftLength = 168000

carrier_FftClass = Classes.qsfFFT(carrier_path, sampleRate, carrAudioLengthSamp, freqCoef)
carrier_Fft = carrier_FftClass.rfft(brickWallHiPass)
carrier_fft_Plot = carrier_FftClass.fftPlotNorm()

carrierFft_Size = carrier_FftClass.size_rfft()

sourceOBJ_convo = np.zeros((numObjects, carrierFft_Size), dtype=np.complex128)

numConvoBlocks = math.ceil(carrierFft_Size / impulse_rfftSize)

print(numConvoBlocks)

breakpoint()

print(len(sourceOBJ_convo[0]))
for i in range(numObjects):
    sourceOBJ_convo[i] = (obj_Response[i] * carrier_Fft).copy()

breakpoint()

# Genetic Operators

evoClass = evoOp.NewGeneration(obj_Response, numObjects, impulse_rfftSize, carrier_Fft, freqCoef, populationLimit)

newGen = evoClass.newGen()

newGenNoMutation = evoClass.newGenNoMutation()

newGen_CarrierConvo = evoClass.convolve()

print("Generation 0")

# IRFFT. The inverse fourier transform

carrier_irfft = np.fft.irfft(carrier_Fft)
carrier_irfft = np.real(carrier_irfft)

for i in range(numObjects):
    obj_irfft[i] = np.fft.irfft(obj_rfft[i])[0:impulse_size]
    obj_irfft[i] = np.real(obj_irfft[i])
    obj_carr_convo_irfft[i] = np.fft.irfft(sourceOBJ_convo[i])[0:impulse_size]
    obj_carr_convo_irfft[i] = np.real(obj_carr_convo_irfft[i])
    impulse_irfft[i] = np.fft.irfft(obj_Response[i])[0:impulse_size]
    impulse_irfft[i] = np.real(impulse_irfft[i])

for i in range(numObjects**2 - numObjects):
    newGen_irfft[i] = np.fft.irfft(newGen[i])[0:impulse_size]
    newGen_irfft[i] = np.real(newGen_irfft[i])
    newGen_carr_convo_irfft[i] = np.fft.irfft(newGen_CarrierConvo[i])[0:impulse_size]
    newGen_carr_convo_irfft[i] = np.real(newGen_carr_convo_irfft[i])

# Data visualization

if visualise == True:

    numPlotsPerPage = 6
    numPlots = int((numObjects ** 2 - numObjects) / numPlotsPerPage)
    firstPlot = 0
    lastPlot = numPlotsPerPage - 1
    
    newGenPlot = np.ndarray((numPlots,), dtype=object)
    gen0Plot = qsfPlot.Qsfplot(sampleRate, numObjects, impulse_rfftSize, impulse_size)
    gen0Plot.plotFirstGen(evoClass.getSlice(), objectNames, obj_FftClass)
    
    for i in range(numPlots):
        newGenPlot[i] = qsfPlot.Qsfplot(sampleRate, numPlotsPerPage, impulse_rfftSize, impulse_size)
        newGenPlot[i].plotNewGen(evoClass.getSlice(), objectNames, newGen[firstPlot:lastPlot], newGen_irfft[firstPlot:lastPlot], newGen_carr_convo_irfft[firstPlot:lastPlot], firstPlot, lastPlot)
        # qsfPlot.plotNewGen(evoClass.getSlice(), sampleRate, objectNames, newGen[firstPlot:lastPlot], newGen_irfft[firstPlot:lastPlot], newGen_carr_convo_irfft[firstPlot:lastPlot], firstPlot, lastPlot)
        firstPlot += numPlotsPerPage
        lastPlot += numPlotsPerPage

    for i in range(numPlots):
        newGenPlot[i].showPlot()

    gen0Plot.showPlot()

# breakpoint()

    async def work():
        genIndex = 1
        while True:
            await asyncio.sleep(5)

            newGen = evoClass.nextGen()

            newGenNoMutation = evoClass.newGenNoMutation()

            newGen_CarrierConvo = evoClass.convolve()

            print("Generation " + str(genIndex))
            genIndex += 1

            if visulise == True:

                numPlotsPerPage = 6
                numPlots = int((numObjects ** 2 - numObjects) / numPlotsPerPage)
                firstPlot = 0
                lastPlot = numPlotsPerPage - 1

                newGenPlot = np.ndarray((numPlots,), dtype=object)

                for i in range(numPlots):
                    newGenPlot[i] = qsfPlot.Qsfplot(sampleRate, numPlotsPerPage, impulse_rfftSize, impulse_size)
                    newGenPlot[i].plotNewGen(evoClass.getSlice(), objectNames, newGen[firstPlot:lastPlot],
                                            newGen_irfft[firstPlot:lastPlot], newGen_carr_convo_irfft[firstPlot:lastPlot],
                                            firstPlot, lastPlot)
                    # qsfPlot.plotNewGen(evoClass.getSlice(), sampleRate, objectNames, newGen[firstPlot:lastPlot], newGen_irfft[firstPlot:lastPlot], newGen_carr_convo_irfft[firstPlot:lastPlot], firstPlot, lastPlot)
                    firstPlot += numPlotsPerPage
                    lastPlot += numPlotsPerPage

                for i in range(numPlots):
                    newGenPlot[i].showPlot()

                gen0Plot.showPlot()

# print(len_data)

# Create a folder for storing audio renders, identified by date
renderID_folder = ''
objectConvolveFolder = ''

if (writeFiles == True):
    renderFolder = output_folder + '/' + date_string
    print(renderFolder)
    breakpoint()
    renderID_folder = renderFolder + '/' + dt_string
    objectConvolveFolder = renderFolder + '/' + 'Object_Carrier_Convolution/'
    objectRawImpulseFolder = renderFolder + '/' + 'Object_Raw_Impulse/'
    if not os.path.exists(renderFolder):
        os.makedirs(renderFolder)

    if not os.path.exists(objectConvolveFolder):
        os.makedirs(objectConvolveFolder)

    if not os.path.exists(objectConvolveFolder):
        os.makedirs(objectConvolveFolder)

    if not os.path.exists(objectRawImpulseFolder):
        os.makedirs(objectRawImpulseFolder)

    if genID == 0:
        for i in range(numObjects):
            objectConcolvePath = objectConvolveFolder + objectNames[i] + '.wav'
            objectImpulsePath = objectRawImpulseFolder + objectNames[i] + 'impulse' + '.wav'

            if not os.path.exists(objectConcolvePath):
                scipy.io.wavfile.write(objectConcolvePath, sampleRate, librosa.util.normalize(obj_carr_convo_irfft[i]))

            if not os.path.exists(objectImpulsePath):
                scipy.io.wavfile.write(objectImpulsePath, sampleRate, librosa.util.normalize(impulse_irfft[i]))

    for i in range(numObjects):
        newGenImpulsePath = renderID_folder + '/New Generation Impulse_' + dt_string + '_' + str(i) + '_' + str((i + 1) % numObjects) + '.wav'
        newGenConvolutionPath = renderID_folder + '/New Generation Convolution_' + dt_string + '_' + str(i) + '_' + str((i + 1) % numObjects) + '.wav'

        scipy.io.wavfile.write(newGenImpulsePath, sampleRate, librosa.util.normalize(newGen_irfft[i]))
        scipy.io.wavfile.write(newGenConvolutionPath, sampleRate, librosa.util.normalize(newGen_carr_convo_irfft[i]))

if (run_evolution == True):
    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(work())
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        call(["open", renderID_folder])
        call(["open", objectConvolveFolder])
        loop.close()


