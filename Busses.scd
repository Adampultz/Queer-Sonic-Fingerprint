// Busses for real-time control

~convolveMix = ~numEnvironments.collect{Bus.control(s).set(-1.0)}; // Wet-dry convolution mix. Convolution operates in the feedback chain. -1 = off, 1 = fully convolved. If using a nanoKontrol, mapped to dial 2

~feedbackVol = ~numEnvironments.collect{Bus.control(s)}; // 0 - 1. Controls the gain of the microphone. NanoKontrol: slider 2

~feedbackThreshBus = ~numEnvironments.collect{Bus.control(s).set(0.01)}; // 0 - 1. Controls the gain of the microphone. NanoKontrol: slider 2

~irVolume = ~numEnvironments.collect{Bus.control(s)}; // 0 - 1. Volume of the playback of the irs. NanoKontrol: slider 3

~gyroBus = ~numSensors.collect{|i| 3.collect{ Bus.control(s)}};
~accBus = ~numSensors.collect{|i| 3.collect{ Bus.control(s)}};

~feedbackAMfreq = ~numSensors.collect{|i| Bus.control(s).set(0.1)};

~feedbackSensorAmp = ~numSensors.collect{|i| Bus.control(s).set(1)};

~conversationVolume = Bus.control(s);

~conversationConvMix = Bus.control(s).set(0);

~masterVol = Bus.control(s); // 0 -1. Master volume. NanoKontrol: slider 1

~convolveAmp = Bus.control(s).set(0.5); // 0 -1. Master volume. NanoKontrol: slider 1

~mutationProbMultiply = Bus.control(s).set(1); // 0 -1. Master volume. NanoKontrol: slider 1

~convPanBus = Bus.control(s).set(0); // 0 -1. Master volume. NanoKontrol: slider 1

~cleanFilterBus = Bus.control(s).set(-1);

// Analysis busses

~envCent = ~numEnvironments.collect{Bus.control(s)};

~envCentAvg = ~numEnvironments.collect{Bus.control(s)};

~envSpread = ~numEnvironments.collect{Bus.control(s)};

~distanceBus = ~numEnvironments.collect{~maxNumIndividuals.collect{Bus.control(s)}};

~specCentAvgBus = ~numEnvironments.collect{~maxNumIndividuals.collect{Bus.control(s)}};

~envirIrAmpBus = ~numEnvironments.collect{Bus.control(s).set(0.3)};

// Audio busses

~mixerBus = Bus.audio(s, numChannels: ~numEnvironments);

~exciterBus = ~numExciters.collect{Bus.audio(s, numChannels: 1)};
