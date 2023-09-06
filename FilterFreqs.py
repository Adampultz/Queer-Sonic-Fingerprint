
# freqIndex = np.where(pitches)[0] / freqCoeff
freqDist = 0
prevFreq = 0
iterInd = 0
amps = []
firstRun = True

centreFreqs = []
freqMinDist = 10

for i in range(len(convolve_abs)):
    if (convolve_abs[i] > 0.5):
        thisFreq = i / freqCoeff
        amp = convolve_abs[i]
        if (firstRun == True):
            freqCluster = [thisFreq]
            ampCluster = [amp]
            prevFreq = thisFreq
            firstRun = False
        if (thisFreq - prevFreq <= freqMinDist):
            freqCluster.append(thisFreq)
            ampCluster.append(amp)
            prevFreq = thisFreq
        else:
            clusterLen = len(freqCluster)
            if (iterInd == 0):
                freqIndex = [freqCluster]
                ampIndex = [ampCluster]
                centreFreqs = [sum(freqCluster) / len(freqCluster)]
                bandWidths = [freqCluster[clusterLen - 1] - freqCluster[0]]
                amps = [sum(ampCluster)]
                iterInd = 1
            else:
                freqIndex.append(freqCluster)
                ampIndex.append(ampCluster)
                if (len(freqCluster) > 0):
                    centreFreqs.append(sum(freqCluster) / len(freqCluster))
                    bandWidths.append(freqCluster[clusterLen - 1] - freqCluster[0])
                    amps.append(sum(ampCluster))
                else:
                    centreFreqs.append(0)
                    bandWidths.append(0)
                    amps.append(0)
            freqCluster = []
            ampCluster = []
        prevFreq = thisFreq


if (sendOsc == True):
    for i in range(len(centreFreqs)):
        length = len(centreFreqs)
        freqs = centreFreqs[i]
        filterAmps = ampsNorm[i]
        filterBw = bandWidths[i]
        client.send_message("/filter", [length, i, freqs, filterBw, filterAmps])

# print(len(freqIndex))
# print(amps[15:18])


# ampsNorm = librosa.util.normalize(amps)

# ampsNorm = numpy.zeros(61)

# for i in range(len(amps)):
#     print(amps[i])
#     # ampsNorm[i] = librosa.util.normalize(amps[i])
#     ampsNorm[i] = 0.5
#     # print(ampsNorm[i])

# print(ampsNorm)

# tempFreqArray = []
# tempAmpArray = []
# centreFreqArray = []
# bwArray = []
# prevFreq = freqIndex[0]
# for (i in freqIndex):
#     currentFreq = freqIndex[i]
#     if (currentFreq - prevFreq) < freqMinDist:
#         tempFreqArray.append(currentFreq)
#         tempAmpArray.append()
#         prevFreq = currentFreq
#     else:



# print(freqIndex[:200])

# print(len(freqIndex[30:len(freqIndex)]))


# freqAmps =

# resFreqs = []
#
# for i in pitches:
#     print(pitches[i])