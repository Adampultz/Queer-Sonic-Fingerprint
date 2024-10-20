~sensorOSCdefs = ~numSensors.collect{|n|
	OSCdef("\sensorOsc_%".format(n), {|msg, time, addr, recvPort|
	// if(msg[1] < ~sensorMin[0][0], {~sensorMin[0][0] = msg[1]})

		if (~oscMinMaxTrace,
	~numSensorVals.do{|i|
		var val = msg[i + 1];
		if(val > ~sensorMax[n][i], {~sensorMax[n][i] = val
		});
			if(val < ~sensorMin[n][i], {~sensorMin[n][i] = val})};
		);

		3.do{|k|
			var val = msg[k + 1];
			~gyroBus[n][k].set(val.linlin(~sensorMin[n][k], ~sensorMax[n][k], 0.0, 1.0))};

		3.do{|k|
			var val = msg[k + 4];
			~accBus[n][k].set(val.linlin(~sensorMin[n][k + 3], ~sensorMax[n][k + 3], 0.0, 1.0))};

	}, "/ESP32_%".format(n);, ~netAddresses[n]); // def style
};
