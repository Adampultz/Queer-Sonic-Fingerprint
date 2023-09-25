import numpy as np
from numpy.random import default_rng
import random

class NewGeneration:
    def __init__(self, obj_Response, numObjects, impulse_rfftSize, carrier_Fft, freqCoeff, populationMax):
        self.impulse_rfftSize_ = impulse_rfftSize
        self.slice_ = 0
        self.slicePoint_ = 0
        self.numObjects_ = numObjects
        self.padding_ = 0.10
        self.bias_ = 0
        self.obj_Response_ = obj_Response
        self.carrier_Fft_ = carrier_Fft
        self.freqCoeff_ = freqCoeff
        self.popMax = populationMax
        self.popmaxhalf = int(self.popMax / 2)
        self.populationSize = numObjects**2 - numObjects
        self.popindexes = np.arange(0, populationMax, 1)
        self.currentPopSize = numObjects
        self.popgoalincrease = self.popMax / self.currentPopSize
        self.replacementcoeff = 0.5 # How many individuals are replaced every generation (decimal fraction)
        self.matingcoeff = 1 / self.replacementcoeff # How many times does each individual mate (each mating produces one offspring)
        self.popincrease = 1 # 1 = stable, < 1 = decreasing, > 1 = increasing

        self.newGen_CarrierConvo_ = np.zeros((populationMax, impulse_rfftSize), dtype=np.complex128)
        self.prevGen_ = np.zeros((populationMax, impulse_rfftSize), dtype=np.complex128)
        self.children_ = np.zeros((populationMax, 2, impulse_rfftSize), dtype=np.complex128)
        self.newGen_ = np.ndarray((populationMax, impulse_rfftSize), dtype=np.complex128)
        self.newGenMutation_ = np.zeros((populationMax, impulse_rfftSize), dtype=np.complex128)
        self.newpopsel = np.zeros((populationMax, 3), dtype=int)

    def newGen(self):
        # To do: put slice point inside the for loop
        self.slice()

        for n in range(self.numObjects_):
            self.children_[n][0][0:self.slicePoint_] = self.obj_Response_[n][0:self.slicePoint_].copy()
            self.children_[n][1][
                              self.slicePoint_:self.impulse_rfftSize_] = self.obj_Response_[(n + 1) % self.numObjects_][
                              self.slicePoint_:self.impulse_rfftSize_].copy()
        genIndex = 0

        for n in range(self.numObjects_):
            for k in range(self.numObjects_ - 1):
                self.newGen_[genIndex] = np.concatenate((self.children_[n][0][0:self.slicePoint_], self.children_[(n + k + 1) % self.numObjects_][1][
                              self.slicePoint_:self.impulse_rfftSize_]))

                self.newGenMutation_[genIndex] = self.mutation(self.newGen_[genIndex].copy(), genIndex, 1, 20)
                genIndex += 1

        self.currentPopSize = genIndex

        return self.newGenMutation_[0:self.populationSize]

    def nextGen(self):
        self.slice()

        selParents = self.selection()

        for i in range(int(self.currentPopSize * self.replacementcoeff)):
            parent_0 = selParents[i][0]

            for n in range(2):
                newgenindex = (i * 2) + n
                parent_1 = selParents[i][n + 1]
                child_1 = self.newGenMutation_[parent_0][0:self.slicePoint_].copy()
                child_2 = self.newGenMutation_[parent_1][self.slicePoint_:self.impulse_rfftSize_].copy()
                self.newGen_[i] = np.concatenate((child_1, child_2))
                self.newGenMutation_[i] = self.mutation(self.newGen_[i].copy(), i, 1, 20)
                print(newgenindex)
            child_1 = self.newGenMutation_[i][0:self.slicePoint_].copy()
            child_2 = self.newGenMutation_[(i + 1) % 2][self.slicePoint_:self.impulse_rfftSize_].copy()
            self.newGen_[i] = np.concatenate((child_1, child_2))
            self.newGenMutation_[i] = self.mutation(self.newGen_[i].copy(), i, 1, 20)

        return self.newGenMutation_

    def convolve(self):
        for i in range(self.populationSize):
            self.newGen_CarrierConvo_[i] = (self.newGenMutation_[i] * self.carrier_Fft_)

        return self.newGen_CarrierConvo_[0:self.populationSize]

    def newGenNoMutation(self):
        return self.newGen_[0:self.populationSize]

    def slice(self):
        padding = self.padding_ * self.impulse_rfftSize_
        self.slice_ = np.random.default_rng().integers(low=0, high=self.impulse_rfftSize_)
        self.slicePoint_ = int(self.slice_ * self.freqCoeff_)

    def getSlice(self):
        return self.slice_

    def slicePoint(self):
        return self.slicePoint_

    def selection(self):
        weights = np.random.random(self.currentPopSize) # For prototyping: Create array of random weights
        # Scale so that all weights sum to 1
        sum = np.sum(weights)
        weights = weights / sum

        # Select array of first parents.
        parent_1 = np.random.choice(self.popindexes[0:self.currentPopSize], int(self.currentPopSize * self.replacementcoeff), False, weights)

        # Choose two mates for each parent
        for i in range(int(self.currentPopSize * self.replacementcoeff)):
            self.newpopsel[i][0] = parent_1[i] # Adds initial parent as first index
            mates = np.random.choice(self.popindexes[0:self.currentPopSize], 2, False, weights) # Choose two mates
            # Add each mate to second and third index of each parent
            for n in range(2):
                self.newpopsel[i][1 + n] = mates[n]

        return self.newpopsel[0:self.currentPopSize]

    def mutation(self, newGenFFT, mutationindex, mutationFactor, frequencybleed):
        probability = (1 / self.impulse_rfftSize_) * mutationFactor
        mutations = np.zeros((10 * 10,))
        index = 0
        bleed = frequencybleed * self.freqCoeff_
        halfBleed = bleed / 2
        amplitude = 100
        envelope = np.hamming(bleed)
        invEnvelope = 1 - envelope
        upperLimit = int(self.impulse_rfftSize_ - halfBleed)

        for i in range(self.impulse_rfftSize_):

            while (i >= 20) and (i <= upperLimit):
                mutation = random.choices([0,1], weights=[1 - probability, probability])

                if mutation[0] == 1:
                    fftIndex = int(i * self.freqCoeff_)
                    # print('Mutation in ' + str(mutationindex) + ' at ' + str(i / self.freqCoeff_))
                    mutations[index] = i
                    index += 1

                    for n in range(int(bleed)):
                        bleedIndex = int(i - (halfBleed - n))
                        newGenFFT[bleedIndex] *= (1 + envelope[n]) * 10
                break

        return newGenFFT