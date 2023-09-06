import numpy as np
import librosa

class qsfFFT:
  def __init__(self, audio, sRate, length, carrAudioLength):
    aud_data, rate = librosa.load(audio, sr = sRate)
    aud_data = aud_data[0:length]

    if (np.max(aud_data) < 1):
        aud_data = librosa.util.normalize(aud_data)

    self.carrAudioLength = carrAudioLength
    self.audioData = aud_data
    self.audioDataArray = np.asarray(aud_data, dtype = np.float64)
    self.sRate = rate

  def audio(self):
      return self.audioData

  def fft(self, loBrickWall): # perform fft, with brickwall hipass and DC filters
      data = self.audioData
      size = len(data)
      array = data[0: size]

      if (size < self.carrAudioLength):
        array = np.pad(array, (0, self.carrAudioLength - size), 'constant')

      fourier = np.fft.fft(array)
      fourier = fourier - np.mean(fourier) # DC offset

      for i in range(loBrickWall):
          fourier[i] = 0

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




