s.options.memSize_(2.pow(20));
s.options.sampleRate_(48000);
s.options.numInputBusChannels_(6);
s.options.numOutputBusChannels_(40);
s.options.hardwareBufferSize = 512;
s.options.maxSynthDefs = 2048;
s.options.maxNodes = 2048;
s.recHeaderFormat = "wav";
s.options.numBuffers = 4096 * 2;

s.options.outDevice_(
	// "MacBook Pro Speakers"
	// "External Headphones"
	// "ZoomAudioD"
	"Fireface UCX (23732049)"
	// "Fireface + Blackhole"
	// "Blackhole + Computer Audio"
	// "Fireface + Zoom"
	// "Zoom + Fireface"
	// "Scarlett 2i2 USB"
	//"Komplete Audio 6"
);

s.options.inDevice_(
	// "MacBook Pro Microphone"
	"Fireface UCX (23732049)"
	// "ZoomAudioD"
	// "Fireface + Blackhole"
	// "Blackhole + Computer Audio"
	// "Fireface + Zoom"
	// "Zoom + Fireface"
	// "Scarlett 2i2 USB"x
	//"Komplete Audio 6"
);

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

	/*	~objPath = "/Users/adammac2023/Documents/Musik-business/Adam/2024/Speculative Sound Synthesis/Code_27.07.2024/Impulse_responses/";*/

	~objPath = "/Users/adammac2023/Documents/Musik-business/Projects/Queer Sonic Fingerprint/Audio/Impulse_Responses/QSF_ALB/";

	~museumPath = "/Users/adammac2023/Documents/Musik-business/Projects/Queer Sonic Fingerprint/Audio/Museum Audio/";

	~test = PathName(~objPath).files.do{|i|};

	~numspeakerSqrt = ~numEnvironments.sqrt;

	"Initialising conversation buffers".postln;
	~conversationBuffers = ~readFolderOfFolders.(~museumPath);
	s.sync;
	"Conversation buffers initialised".postln;

	~parentBuffers = ~test.size.collect{|i| Signal.read(~test[i].fullPath)};

	~cutPoints = [0, 20, 28, 36, 44];

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

	// ~parentIRs = ~newSigBufs.deepCopy;

	s.sync;

	/*~convolveSpectrum = ~prepConvolve.(~newSigBufs.choose, ~convolveFftsize);
	~convolveSpectrum = ~convolveSpectrum.normalize;*/

	s.sync;

	Synth(\mixer, [\amp, ~masterVol], target: ~mixerGrp);
	Synth(\transducerMixer, [\amp, ~masterVol], target: ~mixerGrp);

	~synthSensor0 = Synth(\sensor0, target: ~envGrp);
	~synthSensor1 = Synth(\sensor1, target: ~envGrp);
	~synthSensor2 = Synth(\sensor2, target: ~envGrp);
	~synthSensor3 = Synth(\sensor3, target: ~envGrp);
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
	/*	~populations.(1, ~newSigs[0], ~parentFreqResp.asList);
	"Two".postln;
	s.sync;
	~populations.(2, ~newSigs[0], ~parentFreqResp.asList);
	"Three".postln;
	s.sync;
	~populations.(3, ~newSigs[0], ~parentFreqResp.asList);
	"Four".postln;*/

	"Done".postln;
};
