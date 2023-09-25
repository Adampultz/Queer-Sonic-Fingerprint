import numpy as np
import librosa
import os


class qsf_ImpulseResponses:
    def __init__(self, obj_path):
        self.path = obj_path
        self.obj_paths_array = []

        objectCount = 0

        for subdir, dirs, files in os.walk(obj_path):
            for file in files:
                if file == '.DS_Store':
                    print("Ignoring file '.DS_Store'")
                else:
                    objectCount += 1

        self.numObjects = objectCount

    def numbObjects(self):
        return self.numObjects

    # def frequencyResponses(self):
    # for subdir, dirs, files in os.walk(path):
    #     index = 0
    #     for file in files:
    #         objFileName = ''
    #         path = os.path.join(subdir, file)
    #         for char in range(len(file)):
    #             if file[char + 2] == '_':
    #                 objectNames[index] = objFileName
    #                 break
    #             else:
    #                 objFileName += file[char + 2]
    #
    #         obj_paths_array.append(path)  # local
    #
    #         obj_FftClass[index] = Classes.qsfFFT(path, sampleRate, impulse_rfftSize, carrAudioLengthSamp)
    #         obj_rfft[index] = obj_FftClass[index].rfft(brickWallHiPass)
    #         obj_fft_Plot[index] = obj_FftClass[index].fftPlotNorm()
    #
    #         response = Classes.qsfObjFreqR(obj_rfft[index], impulse_rfft)  # Local
    #         obj_ResponseClass[index] = response
    #         obj_Response.append(response.response())
    #
    #         index += 1


class qsfFFT:
    def __init__(self, audio, sRate, length, carrAudioLength, freqCoef):
        aud_data, rate = librosa.load(audio, sr=sRate)
        aud_data = aud_data[0:length]

        if (np.max(aud_data) < 1):
            aud_data = librosa.util.normalize(aud_data)

        self.carrAudioLength = carrAudioLength
        self.audioData = aud_data
        self.audioDataArray = np.asarray(aud_data, dtype=np.float64)
        self.sRate = rate
        self.freqCoef = freqCoef

    def audio(self):
        return self.audioData

    def fft(self, loBrickWall, DC_filter):  # perform fft, with brickwall hipass and DC filters
        data = self.audioData
        size = len(data)
        array = data[0: size]

        if (size < self.carrAudioLength):
            array = np.pad(array, (0, self.carrAudioLength - size), 'constant')

        fourier = np.fft.fft(array)

        if DC_filter == 1:
            fourier = fourier - np.mean(fourier) # DC offset

        for i in range(int(loBrickWall * self.freqCoef)):
            fourier[i] *= 0

        self.rfft = fourier
        return fourier

# perform rfft, with brickwall hipass and DC filters. Rfft only produce the first half of the fft transform, as the second half is a mirror image
    def rfft(self, loBrickWall):
        data = self.audioData
        size = len(data)
        array = data[0: size]

        if (size < self.carrAudioLength):
            array = np.pad(array, (0, self.carrAudioLength - size), 'constant')

        fourier = np.fft.rfft(array)
        fourier = fourier - np.mean(fourier) # DC offset

        for i in range(loBrickWall):
            fourier[i] = 0
        self.rfft = fourier

        return fourier

    def size_rfft(self):
        if hasattr(self, 'rfft'):
            return len(self.rfft)
        else:
            print("Please calculate rfft before calling size")
        return 0

    def fftPlot(self):
        if hasattr(self, 'rfft'):
            pass
        else:
            self.rfft()

        fourier_to_plot_abs = np.abs(self.rfft)
        self.fftPlotAbs = fourier_to_plot_abs

        return fourier_to_plot_abs

    def fftPlotNorm(self):
        if hasattr(self, 'fftPlotAbs'):
            pass
        else:
            self.fftPlot()

        fourier_to_plot_absNorm = librosa.util.normalize(self.fftPlotAbs)
        return fourier_to_plot_absNorm

# Frequency response of object. Takes the fft of the frequency response of an object and divides it by the frequency response
# of the impulse used to generate the object's frequency response

class qsfObjFreqR:
    def __init__(self, object, sine):
        self.division = object / sine
        self.abs = abs(self.division)

    def response(self):
        return self.division

    def absNorm(self):
        return librosa.util.normalize(self.abs)




