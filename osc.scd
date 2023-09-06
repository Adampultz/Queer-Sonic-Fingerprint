s.plotTree

OSCFunc.trace(true);
OSCFunc.trace(false);

s.boot;

thisProcess.openUDPPort(64750);

~oscdAddr = NetAddr("127.0.0.1", 64750);

~filterFreqs = Array.fill(61, {|i| 0});

~oscFromPy = OSCFunc({|msg, time, addr, recvPort|
	var index = msg[2];
	msg.postln;
/*	~filterFreqs[2] = msg[3]*/
if(index == 0) {
		~filterFreqs = Array.fill(msg[1], {|i| 0});
		~filterBw = Array.fill(msg[1], {|i| 0});
		~filterAmps = Array.fill(msg[1], {|i| 0});
		~filterFreqs[0] = msg[3];
		~filterBw[0] = msg[4];
		~filterAmps[0] = msg[5];
	} {
		~filterFreqs[index] = msg[3];
		~filterBw[index] = msg[4];
		~filterAmps[index] = msg[5];
	}
}, '/filter')


{var noise = PinkNoise.ar;
	LPF.ar(Mix(Resonz.ar(noise, ~filterFreqs, ~filterBw / ~filterFreqs, ~filterAmps)), 6000)}.play

{WhiteNoise.ar(0.1)}.play

{var noise = Impulse.ar(0.5);
	DynKlank.ar(`[~filterFreqs, ~filterAmps, Array.fill(61, {1})], noise)}.play;