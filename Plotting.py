import matplotlib.pyplot as plt
import Defs as defs
import numpy as np
import librosa

class Qsfplot:
    def __init__(self, sampleRate, numObjects, fftSize, impulseSize):
        self.fftsize = fftSize
        self.impulsesize = impulseSize
        self.sampleRate = sampleRate
        self.numObjects = numObjects
        self.w = np.linspace(0, self.sampleRate // 2, fftSize)
        self.audioPltLn = np.linspace(0, impulseSize // self.sampleRate, impulseSize)

    def plot(self, ):
        if visulise == True:

            numPlotsPerPage = 6
            numPlots = int((numObjects ** 2 - numObjects) / numPlotsPerPage)
            firstPlot = 0
            lastPlot = numPlotsPerPage - 1

            newGenPlot = np.ndarray((numPlots,), dtype=object)

            gen0Plot = qsfPlot.Qsfplot(sampleRate, numObjects, impulse_rfftSize, impulse_size)

            gen0Plot.plotFirstGen(evoClass.getSlice(), objectNames, obj_FftClass)

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

# class Qsfplot:
#     def __init__(self, sampleRate, numObjects, fftSize, impulseSize, numPlotsPerPage):
#         if visulise == True:
#             self.fftsize = fftSize
#             self.impulsesize = impulseSize
#             self.sampleRate = sampleRate
#             self.numObjects = numObjects
#             self.w = np.linspace(0, self.sampleRate // 2, fftSize)
#             self.audioPltLn = np.linspace(0, impulseSize // self.sampleRate, impulseSize)
#
#             numPlotsPerPage = 6
#             numPlots = int((numObjects ** 2 - numObjects) / numPlotsPerPage)
#             firstPlot = 0
#             lastPlot = numPlotsPerPage - 1
#
#             newGenPlot = np.ndarray((numPlots,), dtype=object)
#
#             gen0Plot = qsfPlot.Qsfplot(sampleRate, numObjects, impulse_rfftSize, impulse_size)
#
#             gen0Plot.plotFirstGen(evoClass.getSlice(), objectNames, obj_FftClass)
#
#             for i in range(numPlots):
#                 newGenPlot[i] = qsfPlot.Qsfplot(sampleRate, numPlotsPerPage, impulse_rfftSize, impulse_size)
#                 newGenPlot[i].plotNewGen(evoClass.getSlice(), objectNames, newGen[firstPlot:lastPlot],
#                                          newGen_irfft[firstPlot:lastPlot], newGen_carr_convo_irfft[firstPlot:lastPlot],
#                                          firstPlot, lastPlot)
#                 # qsfPlot.plotNewGen(evoClass.getSlice(), sampleRate, objectNames, newGen[firstPlot:lastPlot], newGen_irfft[firstPlot:lastPlot], newGen_carr_convo_irfft[firstPlot:lastPlot], firstPlot, lastPlot)
#                 firstPlot += numPlotsPerPage
#                 lastPlot += numPlotsPerPage
#
#             for i in range(numPlots):
#                 newGenPlot[i].showPlot()
#
#             gen0Plot.showPlot()

    def plotFirstGen(self, slice, objectNames, obj_FftClass):

        numColumns = 2
        numRows = self.numObjects

        axisArray = np.ndarray((numRows, numColumns), dtype=object)
        obj_fft_Plot = np.zeros((self.numObjects, self.fftsize))


        for i in range(self.numObjects):
            obj_fft_Plot[i] = obj_FftClass[i].fftPlotNorm()

        fig, axs = plt.subplots(numRows, numColumns, figsize=(20, 10))

        # Scale slice point to fit in visualisation
        scaledSlice = defs.scale_number(slice, 0, self.fftsize, 0, self.sampleRate / 2)

        for row in range(numRows):
            objID = row
            plotRow = (row)

            column = 0
            column_2 = 1
            column_3 = 2
            column_4 = 3
            print(objectNames[objID])
            axisArray[plotRow][column] = axs[plotRow, column]
            axisArray[plotRow][column].plot(self.audioPltLn,  obj_FftClass[objID].audio())
            axisArray[plotRow][column].set_ylabel('Amplitude')
            axisArray[plotRow][column].title.set_text(objectNames[objID] + ' impulse response (time domain)')

            axisArray[plotRow][column_2] = axs[plotRow, column_2]
            axisArray[plotRow][column_2].plot(self.w, obj_fft_Plot[objID])
            axisArray[plotRow][column_2].vlines(x=scaledSlice, ymin=0, ymax=0.5,
                                   colors = 'red',
                                   label = 'slice point')
            axisArray[plotRow][column_2].title.set_text(objectNames[objID] + ' frequency response (frequency domain)')

    def plotNewGen(self, slice, objectNames, FFT, impulseResponse, audioConvolve,
                       firstPlot, lastPlot):
        numObjects_ = len(FFT)
        numColumns = 3
        numRows = numObjects_
        FFT_ = FFT.copy()
        impulseResponse_ = impulseResponse.copy()
        audioConvolve_ = audioConvolve.copy()

        axisArray = np.ndarray((numRows, numColumns), dtype=object)
        obj_fft_Plot = np.zeros((numObjects_, self.fftsize))


        for i in range(numObjects_):
            obj_fft_Plot[i] = librosa.util.normalize(np.abs(FFT[i]))

        fig, axs = plt.subplots(numRows, numColumns, figsize=(20, 10),
                                    num='New generations from ' + str(firstPlot) + ' to ' + str(lastPlot))

        # plt.figure(num='This is the title')

        # Scale slice point to fit in visualisation
        scaledSlice = defs.scale_number(slice, 0, self.fftsize, 0, self.sampleRate / 2)

        for row in range(numRows):
            objID = row
            plotRow = (row)

            column = 0
            column_2 = 1
            column_3 = 2
            column_4 = 3

            axisArray[plotRow][column] = axs[plotRow, column]
            axisArray[plotRow][column].plot(self.w, obj_fft_Plot[objID])
            axisArray[plotRow][column].set_ylabel('Amplitude')
            axisArray[plotRow][column].title.set_text(objectNames[objID] + ' frequency response (frequency domain)')

            axisArray[plotRow][column_2] = axs[plotRow, column_2]
            axisArray[plotRow][column_2].plot(self.audioPltLn, librosa.util.normalize(impulseResponse_[objID]))
            axisArray[plotRow][column_2].title.set_text(
                    objectNames[objID] + ' frequency response (frequency domain)')

            # axisArray[plotRow][column] = axs[plotRow, column]
            # axisArray[plotRow][column].plot(audioPltLn, obj_FftClass[objID].audio())
            # axisArray[plotRow][column].title.set_text(objectNames[objID] + ' impulse response (time domain)')
            #
            # axisArray[plotRow][column] = axs[plotRow, column]
            # axisArray[plotRow][column].plot(audioPltLn, obj_FftClass[objID].audio())
            # axisArray[plotRow][column].title.set_text(objectNames[objID] + ' audio convolution (time domain)')

            axisArray[plotRow][column_3] = axs[plotRow, column_3]
            axisArray[plotRow][column_3].plot(self.audioPltLn, librosa.util.normalize(audioConvolve_[objID]))
            axisArray[plotRow][column_3].title.set_text(
                    objectNames[objID] + ' frequency response (frequency domain)')

            # axisArray[plotRow][column_4] = axs[plotRow, column_4]
            # axisArray[plotRow][column_4].plot(w, np.abs(newGenNoMutation[objID]))

    def showPlot(self):
        plt.show()


