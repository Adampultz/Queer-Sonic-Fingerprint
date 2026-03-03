s.options.memSize_(2.pow(20));
s.options.sampleRate_(48000);
s.options.numInputBusChannels_(12);
s.options.numOutputBusChannels_(40);
s.options.hardwareBufferSize = 512;
s.options.recChannels_(4);
s.options.maxSynthDefs = 2048;
s.options.maxNodes = 2048;
s.recHeaderFormat = "wav";
s.options.numBuffers = 4096 * 3;

"DeviceSetup.scd".loadRelative;

~nyquist = s.options.sampleRate / 2;

"Variables.scd".loadRelative;

"Arrays.scd".loadRelative;


MIDIIn.connectAll;

// Open UDP ports for OSC messages from sensors
~numSensors.do{|i|
	thisProcess.openUDPPort(3334 + i)};

~netAddresses = ~numSensors.collect{|i| NetAddr(~oscAddresses[i], ~udpPort + i)};

s.waitForBoot{
	"Server booted".postln;

	~envGrp = Group.new;
	~synthGrp = Group.new(~envGrp, \addAfter);
	~convolveGrp = Group.new(~envGrp, \addAfter);
	~mixerGrp = Group.new(~synthGrp, \addAfter);

	"Busses.scd".loadRelative;

	s.sync;

	"OSC.scd".loadRelative;

	"Functions.scd".loadRelative;

	s.sync;

	~accRest.do{|i, k|
		3.do{|n|
			i[n] = ~accBus[k][n].getSynchronous;
	}};

	"Paths.scd".loadRelative;

	~test = PathName(~objPath).files.do{|i|};

	~numspeakerSqrt = ~numEnvironments.sqrt;

	~num_populations_playing = 1;

	"Initialising conversation buffers".postln;
	~conversationBuffers = ~readFolderOfFolders.(~museumPath);
	~conversationBuffers = ~conversationBuffers.flatten(1);
	~conversationBufferWeights = Array.fill(~conversationBuffers.size, {1.0 / ~conversationBuffers.size});
	~converSationBufferIndex = ~conversationBuffers.size.collect{|i| i};
	s.sync;
	"Conversation buffers initialised".postln;

	~parentBuffers = ~test.size.collect{|i| Signal.read(~test[i].fullPath)};

	~cutPoints = [0, 20, 30, 40, 47];

	~maximumSize = ~parentBuffers.maxCollectionSize;

	~rfftParams = ~rfftInfo.(~maximumSize);

	~freqResolution = (((~rfftParams[1]).asInteger) / ~nyquist);

	~audioEnv = Env([0,1,1,0], [0.0, 0.9, 0.1], -4).asSignal(~maximumSize);

	~fourierEnv = ~fftEnv.(20, 20000, -4, ~rfftParams[1]);

	~parentFreqResp = ~parentBuffers.size.collect{|i|
		~freqResponse.(~parentBuffers[i], ~rfftParams[0], ~rfftParams[2]);
	}.scramble;

	~parentFreqResp.do{|i|
		i.real.normalizeMagnitude};

	~newSigs[0] = ~parentFreqResp.size.collect{|i| ~newGeneration.(~parentFreqResp[i], ~rfftParams[2], ~audioEnv)};

	~newSigBufs = ~newSigs[0].size.collect{|i| Buffer.loadCollection(s, ~newSigs[0][i])};
	s.sync;

	Synth(\mixer, [\amp, ~masterVol], target: ~mixerGrp);
	Synth(\transducerMixer, [\amp, ~masterVol], target: ~mixerGrp);

	if (~sensors, {
	~synthSensor0 = Synth(\sensor0, target: ~envGrp);
	~synthSensor1 = Synth(\sensor1, target: ~envGrp);
	~synthSensor2 = Synth(\sensor2, target: ~envGrp);
	~synthSensor3 = Synth(\sensor3, target: ~envGrp);
	});

	s.sync;

	"Populations.scd".loadRelative;
	"MIDI.scd".loadRelative;

	~makeControllers.();

	if (~visuals, {
		"Visualisation.scd".loadRelative;
		~freqResponsevisualise.(~parentFreqResp);
	});

	s.sync;

	"Initialise populations".postln;
	// Initialise the first and so far, only community
	~populations.(0, ~newSigs[0][~cutPoints[0]..(~cutPoints[1] -1 )], ~parentFreqResp.asList[~cutPoints[0]..(~cutPoints[1] -1 )]);
	"One".postln;
	s.sync;

	"Done".postln;
};
