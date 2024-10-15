~sensorOSCdefs = ~numSensors.collect{|n|
	OSCdef("\sensorOsc_%".format(n), {|msg, time, addr, recvPort|
	// if(msg[1] < ~sensorMin[0][0], {~sensorMin[0][0] = msg[1]})

	~numSensorVals.do{|i|
		var val = msg[i + 1];
		if(val > ~sensorMax[0][i], {~sensorMax[0][i] = val
		});
			if(val < ~sensorMin[0][i], {~sensorMin[0][i] = val})};

		3.do{|k|
			var val = msg[k + 1];
			~gyroBus[n][k].set(val.linlin(~sensorMin[n][k], ~sensorMax[n][k], 0.0, 1.0))};

		3.do{|k|
			var val = msg[k + 3];
			~accBus[n][k].set(val.linlin(~sensorMin[n][k], ~sensorMax[n][k], 0.0, 1.0))};

	}, "/ESP32_%".format(n);, ~netAddresses[n]); // def style
};
