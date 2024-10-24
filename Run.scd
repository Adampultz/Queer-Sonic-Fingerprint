MIDIFunc.trace(true);
MIDIFunc.trace(false);

OSCFunc.trace(true);
OSCFunc.trace(false);

s.plotTree;

"Main.scd".loadRelative; // Evaluate me

~tasks[0].start; // Begin playback and evolution
~tasks[0].pause;
~tasks[0].resume;
~tasks[0].stop;
~tasks[0].reset;

(
~postInfo[0] = true;
~postInfo[1] = true;
~postInfo[2] = true;
~postInfo[3] = true;
)

(
~postInfo[0] = false;
~postInfo[1] = false;
~postInfo[2] = false;
~postInfo[3] = false;
)

~masterVol.set(1); // 0 -1. Master volume. NanoKontrol: slider 1

(
~numEnvironments.collect{|i|
	~irVolume[i].set(0.3);
	~feedbackVol[i].set(0.9);
	~feedbackThreshBus[i].set(0.02);
	~convolveMix[i].set(1);
	~envirIrAmpBus[i].set(0.1);
	~conversationVolume.set(1);
	~conversationConvMix.set(-1);
};
~convolveAmp.set(0.1);
	~speechCompressThresh = 0.01;
);

(
~adultAge = 5;
~retirementAge = 10;

~deathProb = 0.5;

~visuals = false;

// Volume controls

// EA variables (see Variables.scd for setting these at compile time

~xOverProb = 0.7; // Probability of crossover (0 - 1)

~mutationOn = true;

~zeroPhaseOn = false;

~multiplyOn = true; // For multiplying the complex spectrum of individuals

~mutationWidth = 0.5;

~ancestorReturn = true;
~ancestorReturnProb = 0.1;

~envirIrOn = true;

// ~mutationProb = 0.001; mutation is currently a function of the size of the population, so this is not in use

~envInsert = true; // For inserting a snippet of the environment into the genome of an individual

~spectralEnvelopeOn = false; // For attenuating the high frequencies of the spectrum
~envRecLength = 2.0;

~mutationCoeff = 0.2;

// Fitness Variables. For balancing the parameters contributing to calculating the fitness function. All values are between 0 and 1
~slicepointCoeff = 1; // Biases toward shorter attacks. Seems to favour darker irs when above 0
~specCentDevGenoCoeff = 1.0; // The deviation of the spectral centroid, calculated from the genome
~specCentDevPhenoCoeff = 1.0; // The deviation of the spectral centroid, calculated from the phenome.
~distanceModerator = 0; // If using the lower distance to the spectral centroid. Currently not in use

// The width of the snippet of the environments spectrum that will be inserted into an individual. 0 is nothing, 1 equals the entire range between cut point and the next value on the bark scale. If set very high, it works like a sampler, inserting recognizable audio into irs.
~envInsertWidth = 0.4;

~bufPlayMaxWaitTime = 1.5; // The maximum time added to the wait time between individual buffers playing
)
