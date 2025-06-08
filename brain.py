import math

class ann:
	"""
	   This class handles all functionality for the brain of the robot.

	   Attributes:
	       	'outputPorts': An array consisting of all the components in the robot that produce the output to be fed into the motors (RobotComp[])
	       	'MAX_PARAMS': The maximum amount of parameters that a specific neuron may have (int)
	       	'INPUTS' = number of inputs from the sensors (int)
			'OUTPUTS' = number of output components (int)
			'Weights' = array of weights corresponding to each neuron in the neural network, from input to output (double[])
			'Params' = array of parameters for each neuron, including bias and gain for sigmoid and period, offset and gain for oscillator (doube[])
			'Types' = an array of the type of neuron from input node to output nodes (String[])
			'state' = the output from each neuron after the activation function (double[])
			'input' = the current array of inputs from the sensors (int[])
	   """
	def __init__(self, components, inputs, outputs, weights, params, type):
		"""
			Constructor for the neural network.

			Initializes parameters.

			Parameters:
				'components': the components that make up the output ports (robotComp[])
				'inputs': the number of input neurons (int)
				'outputs': the number of output neurons (int)
				'weights': array of weights that go from the beginning to the end of the network (double[])
				'params': bias, gain, period and offset for output neurons (double[])
				'type': sigmoid/oscillator/input neuron (String[])

		"""
		#max is either (bias, tau, gain) or (phase offset, period, gain)
		self.MAX_PARAMS = 3

		#Branch Hip1 0 to D9
		D9 = 0
		#Branch Hip2 0 to D10
		D10 = 0
		#Branch Hip3 0 to D5
		D5 = 0
		#Branch Knee2 0 to D6
		D6 = 0
		#Branch myid1001 0 to D11
		D11 = 0
		#Branch myid1003 0 to D13
		D13 = 0
		#Branch myid1007 0 to ROLL
		ROLL = 0
		#Branch myid1018 0 to PITCH
		PITCH = 0
		self.outputPorts = []

		for i in components:
			self.outputPorts.append(i)

		NB_LIGHTSENSORS = 0
		NB_TOUCH_SENSORS = 0
		NB_IR_SENSORS = 0
		NB_SERVO_MOTORS = outputs
		NB_ROTATION_MOTORS = 0
		NB_ACC_GYRO_SENSORS = inputs

		#input ports: 0 for lightSensor,
		#type of input: 2 for Accelerometer and Gyroscope
		inputTab = []
		x = [0,2]
		for i in range(inputs):
			inputTab.append(x)

		#FALSE if not irSensor, otherwise index of irSensor
		irIndices = [False, False, False, False, False, False]

		# first arg = value of the output port
		#second arg = type of the output:
		#0 for position control, and 1 for velocity control

		outputTab = []
		for o in range(outputs):
			outputTab.append([self.outputPorts[o],0])

		self.INPUTS = inputs
		self.OUTPUTS = outputs
		self.Weights = weights
		self.Params = params

		#for sigmoid: bias, tau(input or time?), gain
		#for oscilator: period, phase offset (from a central clock), gain
		#oscillator neurons do not receive any input, but rather output a sinusoid oscillation as a function of time
		self.Types = type
		#1 = sigmoid, 3 = oscillator
		self.initNetwork()

	def initNetwork(self):
		"""
			Initializes the input and output states in the neural network to 0.

			The state is the output of each neuron in the network.
		"""
		self.state = []
		#Initialize states
		for o in range(self.OUTPUTS):
			self.state.append(0.0)

		#Initialize inputs
		for i in range(self.INPUTS):
			self.state.append(0.0)

	def feed(self, input):
		"""
			Feeds the network with an array of inputs from the sensors.

			Only initializes the network's current input variable.

			Parameters: 'input': An array of integers that corresponds to the number of input neurons (int[])
		"""
		self.input = []
		print("Creating brain . . .")
		print("Input from sensors: ", input)
		for i in range(self.INPUTS):
			self.input.append(input[i])

	def step(self, time):
		"""
			Feeds the input data into the neural network

			For each input from a sensor, the data is fed into the corresponding neuron and transformed via its activation function.
			This data is propogated through the network to the output layer. The state of each neuron is changed accordingly.

			Parameters: 'time', from the python.time() library, this variable is used to set the state for the oscillator neurons.
		"""
		PI = 3.14159265358979323846
		baseIndexOutputWeights = (self.OUTPUTS)*(self.INPUTS)
		#For each hidden and output neuron, sum the state of all incoming connections
		self.activations = []
		for o in range(self.OUTPUTS):
			self.activations.append(o)
			for i in range(self.INPUTS):
				self.activations[o] = self.activations[o]+self.Weights[(self.OUTPUTS-1)*i+o]*self.state[i]
			for i in range(self.OUTPUTS):
				self.activations[o] = self.activations[o]+self.Weights[baseIndexOutputWeights]+(self.OUTPUTS-1)*i+o*self.state[i]
		#Add in biases and calculate new network state/do appropriate operation for neuron type
		for o in range(self.OUTPUTS):
			#save the next state
			if self.Types[o] == 1:
				#params are bias and gain
				self.activations[o] = self.activations[o]-self.Params[self.MAX_PARAMS*o]
				self.state[o] = (1.0/(1.0 + math.exp(-1*(self.Params[self.MAX_PARAMS*o+1]))))*self.activations[o];
			elif self.Types[o] == 0:
				#linear, params are bias and gain
				self.activations[o] = self.activations[o]-self.Params[self.MAX_PARAMS*o];
				self.state[o] = self.Params[self.MAX_PARAMS*o+1]*self.activations[o];
			elif(self.Types[o] == 3):
				#params are period, phase offset, gain (amplitude)
				period = self.Params[self.MAX_PARAMS * o]
				phaseOffset = self.Params[self.MAX_PARAMS * o + 1];
				gain = self.Params[self.MAX_PARAMS * o + 2]
				self.state[o] = ((math.sin( (2.0 * PI / period)*(time - period * phaseOffset))) + 1.0) / 2.0
				self.state[o] = (0.5 - (gain / 2.0) + self.state[o] * gain)

	def fetch(self):
		"""
			Concatenates the states of each neuron for this forward pass into an array.

			Returns: An array of doubles, control values to be sent to motors (double[])
		"""
		output = []
		for o in range(self.OUTPUTS):
			output.append(self.state[o])
		#returns the output from all output nodes
		print("Output from outputPorts: ", output)
		return output