s.plotTree;

(
s.options.sampleRate_(48000);
s.options.numInputBusChannels_(6);
s.options.numOutputBusChannels_(6);
s.options.hardwareBufferSize = 64;
s.recHeaderFormat = "wav";

s.options.outDevice_(
	 //"Built-in Output"
   "Fireface UCX (23732049)"
	// "Scarlett 2i2 USB"
	//"Komplete Audio 6"
);

s.options.inDevice_(
	"Fireface UCX (23732049)"
	// "Scarlett 2i2 USB"
	//"Komplete Audio 6"
);

~recDur = 3.0;
~padding = 0.5;
~hardwareLatency = 217; // For current macbook + Fireface UCX with buffer size of 64 samples

~sineBuf = Bus.audio(s);

~recPath = PathName(thisProcess.nowExecutingPath.dirname ++ "TestAudio/");

~synthGrp = Group.new;
~recGrp = Group.new(~synthGrp, \addAfter);


s.waitForBoot({

~latencyBuf = 2.collect{Buffer.alloc(s, s.sampleRate * 0.2)};

~recBuf = 2.collect{Buffer.alloc(s, s.sampleRate * (~recDur + ~padding))};

SynthDef(\impulse,
	{|out = 2|
		var in = SoundIn.ar(2);
		var impulse = Impulse.ar(0);
			// var recDirect = RecordBuf.ar(impulse, ~latencyBuf[1], loop: 0, doneAction: 2);
		var recReturn = RecordBuf.ar(impulse + in, ~latencyBuf[1], loop: 0, doneAction: 2);
		Out.ar(2, impulse)
}).add;

SynthDef(\sine,
	{|freqSt = 20, freqEnd = 24000, run = 1, trig = 1|
		var sine, env;
		sine = SinOsc.ar(XLine.kr(freqSt, freqEnd, ~recDur));
		env = EnvGen.kr(Env([0, 1, 1, 0], [0.01, 0.98, 0.01], 'lin'), timeScale: ~recDur, doneAction: 2).scope;
		sine = sine * env;
		Out.ar(~sineBuf, sine);
		Out.ar(2, LeakDC.ar(sine))
}).add;

SynthDef(\rec,
	{|inChan = 0, buf, run = 1, trig = 1|
		var source, sine;
		source = SoundIn.ar(0);
		sine = In.ar(~sineBuf).scope;
		RecordBuf.ar(sine, ~recBuf[0], run: run, loop: 0, trigger: trig, doneAction: 2);
		RecordBuf.ar(source, ~recBuf[1], run: run, loop: 0, trigger: trig, doneAction: 2);
		/*Line.ar(dur: ~recDur + 0.1, doneAction: 2);*/
}).add;

});

)

s.latency

z = Synth(\impulse);

	y = Synth(\sine);

x = Synth(\rec);

(
Routine({
	1.0.wait;
	y = Synth(\sine, target: ~synthGrp);
	x = Synth(\rec, target: ~recGrp);

	(~recDur + 0.5).wait;

	~recBuf[0].normalize.write("/Users/adammac2023/Documents/Musik-business/Projects/Queer Sonic Fingerprint/Code/FFT_Tests_Jan_2022/TestAudio/%".format("kanteenSine_test.wav") , "wav");


	~recBuf[1].normalize.write("/Users/adammac2023/Documents/Musik-business/Projects/Queer Sonic Fingerprint/Code/FFT_Tests_Jan_2022/TestAudio/%".format("kanteenObject_test.wav") , "wav");
}).play;
)


PathName.isAbsolutePathPathName(thisProcess.nowExecutingPath.dirname).pathOnly

PathName(thisProcess.nowExecutingPath.dirname).pathOnly ++ "/TestAudio";



x.free

[~sineSweepBuf, ~objectRecBuf].plot;

~recBuf[0].normalize(-1.0, 1.0).plot;
[~recBuf[0], ~recBuf[1]].plot;
~recBuf[1].play;
~objectRecBuf.play

~objectRecBuf.plot

/*~recBuf[0].write("/Users/adammac2023/Documents/Musik-business/Projects/Queer Sonic Fingerprint/Code/SCImpulseRecord/TestAudio/%".format("sine_Table.wav") , "wav");

~recBuf[1].write("/Users/adammac2023/Documents/Musik-business/Projects/Queer Sonic Fingerprint/Code/SCImpulseRecord/TestAudio/%".format("Table.wav") , "wav");*/

~latencyBuf[1].play

~latencyBuf[0].plot
~latencyBuf[1].plot

~latencyBuf[1].loadToFloatArray(action: {arg array; ~newAudio = array; "done".postln})

~newAudio.size


(
var size = ~newAudio.size;

~adjust = size.collect{|i|
	if (i + ~hardwareLatency <= size, {
		~newAudio[~hardwareLatency + i].postln});
	if (i + ~hardwareLatency > size, {
		0});
};
);

~adjust[8000]

(

var count = 0;
	~newAudio.size.do{ |i|
	var val = ~newAudio[i].round;
	if(count < 2, {
		if(val > 0,{
			i.postln;
			count = count + 1;
		});
	});
};


);

48000 / 2017

~newAudio[1117]

	Env([0, 1, 1, 0], [0.01, 0.98, 0.01], 'exp').plot


	