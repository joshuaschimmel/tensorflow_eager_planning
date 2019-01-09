import numpy

accScale = 10.0
ballXPos = 0.0
ballYPos = 5.0
handXPos = 5.0
handYPos = 0.0
ballVelocity = 1.0
handVelocity = 0.0
numberSteps = 5 # numer of planning steps
acceleration = 1.0*numpy.ones(numberSteps) # random.rand(numberSteps)*accScale #for test/application

#planalgorithm
#first step get the energy outcome
predStateBallXPos = 0.0 # will simply be calculated over the satically ballVelocity
predStateBallYPos = 5.0 # is set statically
predStateHandXPos = 5.0 # is set statically
predStateHandYPos = 0.0 # will be calculated
predStateHandVelocity = 0.0 # will be calculated

class mlp:
	#constructor
	def __init__(self):
		#constants		
		self.valRate = 0.005
		self.valPlanRate = 0.01
		self.numHid = 10
		self.valLearnIterations = 820000
		
		#weights and input
		self.weightsHid = numpy.random.rand(self.numHid,3) - 0.5
		self.weightsOut = numpy.random.rand(2,self.numHid) - 0.5
		
		#bias values for hid and out
		self.biasHid = numpy.zeros(self.numHid)
		self.biasOut = numpy.zeros(2)

	def trans(self,x):
		return 1/(1 + numpy.exp(-x)) 
	
	def feedforward(self, inputs):
		valHid = numpy.dot(self.weightsHid, inputs) + self.biasHid
		self.actHid = self.trans(valHid)
                # linear output with a bias
		actOut = numpy.dot(self.weightsOut, self.actHid)+self.biasOut
		return actOut
		
        # use feedforward() and do backpropagation and learn the forward model
	def iterate(self):
		mean_error = 0.0
                # adjust weights
		for i in range(self.valLearnIterations):

			self.input_pos = (numpy.random.rand(1)*9.0 - 1.0)[0]
			self.input_vel = (numpy.random.rand(1)*8.0 - 3.0)[0]
			self.input_acc = (numpy.random.rand(1)*20.0 - 5.0)[0]
			self.input = numpy.array((self.input_pos, self.input_vel, self.input_acc))

			#fobj = open("out1.txt", "a")
			#fobj.write("%.2f %.2f %.2f\n" % (self.input_pos, self.input_vel, self.input_acc))
			#fobj.close()

			actOut = self.feedforward(self.input)

                        #the physics
			self.phys_pos = self.input_pos + self.input_vel
			self.phys_vel = self.input_vel + self.input_acc / accScale
			self.target = numpy.array((self.phys_pos, self.phys_vel))

			error = self.target - actOut

			mean_error += error**2
			if  i % 10000 == 0:
                            print("mean_sq_error", numpy.sum(mean_error)/10000.0)
                            mean_error = 0.0

                        # backpropagate error
			errorHid = self.actHid*(1-self.actHid) * numpy.dot(numpy.transpose(self.weightsOut),error)

                        # learning
			self.weightsOut = self.weightsOut + self.valRate * numpy.outer(error,self.actHid)
			self.weightsHid = self.weightsHid + self.valRate * numpy.outer(errorHid,self.input)
			
			self.biasHid = self.biasHid + self.valRate * errorHid
			self.biasOut = self.biasOut + self.valRate * error
			
        # use the model for planning
	def backPropPlan(self,errParamQuad, inputs, accelerations):

		#feed forward		
		retAcc = accelerations

		# network activation vectors for all time steps
		valHidB = numpy.zeros((numberSteps,self.numHid))
		actHidB = numpy.zeros((numberSteps,self.numHid))
		actOutB = numpy.zeros((numberSteps,2))

                # do feedforward (rewritten to allow to memorize activations in time, for backprop through time) 
		valHidB[0] = numpy.dot(self.weightsHid, inputs) + self.biasHid
		actHidB[0] = self.trans(valHidB[0])
		actOutB[0] = numpy.dot(self.weightsOut, actHidB[0])+self.biasOut
		
		for i in range(1, numberSteps, 1):
			inputs[0] = actOutB[i-1][0]
			inputs[1] = actOutB[i-1][1]
			inputs[2] = accelerations[i]    #!!!
			valHidB[i] = numpy.dot(self.weightsHid, inputs) + self.biasHid
			actHidB[i] = self.trans(valHidB[i])
			actOutB[i] = numpy.dot(self.weightsOut, actHidB[i])+self.biasOut

                #feed backward			
		eFirst = numpy.zeros((numberSteps,2))
		eSec = numpy.zeros((numberSteps,self.numHid))
		eThird = numpy.zeros((numberSteps,3))

		errParamQuad = numpy.array([5.0 - actOutB[4][0], 0.0])

		eFirst[4] = errParamQuad
		eSec[4]   = actHidB[4]*(1-actHidB[4]) * numpy.dot(numpy.transpose(self.weightsOut),eFirst[4])
		eThird[4] = numpy.dot(numpy.transpose(self.weightsHid),eSec[4])
		retAcc[4] = retAcc[4] + (self.valPlanRate * eThird[4])[2]

		for i in range(numberSteps-2, 0, -1):
			eFirst[i] = numpy.array(eThird[i+1][0],eThird[i+1][1])
			eSec[i]   = actHidB[i]*(1-actHidB[i]) * numpy.dot(numpy.transpose(self.weightsOut),eFirst[i])
			eThird[i] = numpy.dot(numpy.transpose(self.weightsHid),eSec[i])
			retAcc[i] = retAcc[i] + (self.valPlanRate * eSec[i])[2]

		return retAcc

MLP1 = mlp()
print("Network training begins")
MLP1.iterate()
print("Network training finished")

# do the planning
for iterVal in range(3000):
	predStateBallXPos = 0.0 # will simply be calculated over the satically ballVelocity
	predStateBallYPos = 5.0 # is set statically
	predStateHandXPos = 5.0 # is set statically
	predStateHandYPos = 0.0 # will be calculated
	predStateHandVelocity = 0.0 # will be calculated
	
	print("starting planning steps")
	for i in range(numberSteps):
		#feedforward
		print(">>>>>>>")
		inputs = numpy.array((predStateHandYPos, predStateHandVelocity, acceleration[i]))
		predStateHandYPos, predStateHandVelocity = MLP1.feedforward(inputs) #we are using first the input of the acceleration
		
		if iterVal % 10 == 0:
			fobj = open("out.dat", "a")
			fobj.write("%.2f %.2f %.2f\n" % (predStateHandYPos, predStateHandVelocity, acceleration[i]))
			fobj.close()

		print("inputs:", inputs)
		print("outputs:", predStateHandYPos, predStateHandVelocity)
	
		predStateBallXPos += ballVelocity #the ball
		print("<<<<<<<")
		
        # get complete acceleration plan
	acceleration = MLP1.backPropPlan(numpy.array([5.0-predStateHandYPos,0.0]), numpy.array((0,0,acceleration[0])), acceleration)


#move the ball and the hand
for i in range(numberSteps):
	print(i)
	handYPos += handVelocity
	ballXPos += ballVelocity
	handVelocity += acceleration[i]/accScale
	print("-Xpos Ypos vel---")
	print(ballXPos, ballYPos)
	print(handXPos, handYPos, handVelocity)
	print("----")
	print(predStateHandXPos,predStateHandYPos, predStateHandVelocity)

print("======")
print(handYPos-ballYPos)
print(predStateHandYPos-handYPos)

print("Plot the planned trajectories from file using:")
print("gnuplot")
print("set parametric")
print("splot \"out.dat\"")


