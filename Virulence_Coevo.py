# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:15:56 2021

@author: camer
"""

#%% Import packages

from matplotlib.colors import ListedColormap
import numpy as np
from scipy.optimize import root
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.integrate import solve_ivp
import pickle
import scipy.linalg as lin
from datetime import datetime
import pandas as pd
from pathlib import Path
import os
import imageio
from scipy.ndimage.filters import gaussian_filter

#%% Create a class for a simulation

class Simulation:
	'''
	A class which will contain all functions to do with the running, plotting and saving of a simulation.
	'''

	# Initialise and set default values
	def __init__(self,
			  a = 1.0,
			  q = 0.25,
			  b = 0.25,
			  alpha1 = 0.1,
			  alpha2 = 0.1,
			  beta1 = 2.0,
			  c1 = 0.25,
			  c2 = 2.0,
              c3 = 1.0,
			  d = 2.0,
			  delta = 0,
			  gamma1 = 0.05,
			  gamma2 = 0.05,
			  nTraity = 51,
              nTraitb2 = 51,
			  b2traitMin = 0.5,
              b2traitMax = 10.0,
			  ytraitMin = 0.0,
			  ytraitMax = 1.0,
			  yInit = 0.0,
              b2Init = True,
			  extinct = 1e-4,
			  state0 = np.array([0.97,0.01,0.01,0.01]),
			  ntEco = 100,
			  TEco = 100,
			  ntEvo = 2000):

        # Save each of these to the class. Square brackets denote dimension. [T] is the dimension of time and [N] is the dimension of numbers of hosts. If it equals 1, this is a dimensionless parameter.

		# Parameters
		self.a = a  # Growth rate of population. [a] = 1/[T]
		self.q = q  # Density dependent parameter. [q] = 1/([T][N])
		self.b = b  # Natural mortality rate. [b] = 1/[T]
		self.alpha1 = alpha1  # Virulence parameter for mutualist. [alpha1] = 1/[T]
		self.alpha2 = alpha2  # Virulence parameter for parasite. [alpha2] = 1/[T]
		self.beta1 = beta1  # Transmission parameter for mutualist. [beta1] = 1/([T][N])
		self.c1 = c1  # y-cost function parameter. Lies in [0,1]. [c1] = 1
		self.c2 = c2  # y-cost function parameter. [c2] = 1
		self.c3 = c3  # Virulence cost for mutualist. [c3] = 1
		self.d = d  # Power-law for the b2 parameter. [d] = 1
		self.delta = delta  # Switches between tolerance (delta=0) and resistance (delta=1). [delta] = 1
		self.gamma1 = gamma1  # Recovery rate from mutualist. [gamma1] = 1/[T]
		self.gamma2 = gamma2  # Recovery rate from parasite. [gamma2] = 1/[T]    

		# Traits
		self.nTraity = nTraity  # Number of trait values in the y-discretisation.  [nTraity] = 1
		self.nTraitb2 = nTraitb2  # Number of trait values in the b2-discretisation.
		self.ytraitMin = ytraitMin  # Minimum value of ytrait range. [ytraitMin] = 1  [nTraitb2] = 1
		self.b2traitMin = b2traitMin  # Minimum value of b2 trait range. [b2traitMin] = 1
		self.ytraitMax = ytraitMax  # Maximum value of ytrait range. [ytraitMax] = 1
		self.b2traitMax = b2traitMax  # Maximum value of b2-trait range. [b2traitMax] = 1
		self.yInit = yInit  # Initial value for the y trait. [yInit] = 1
		if b2Init == True:  # Use the evolved parasite value
			self.b2Init = ((self.b + self.gamma2 + self.alpha2)/(self.alpha2*(d-1)))**(1/d)  # Initial value for the b2-trait. [b2Init] = 1
		else:
			self.b2Init = b2Init
		self.extinct = extinct  # Extinction threshold for frequency of a trait. [extinct] = 1
		self.yTraits = np.linspace(ytraitMin,ytraitMax,nTraity)  # Discretised y-trait space. Each [traits] = 1
		self.b2Traits = np.linspace(b2traitMin,b2traitMax,nTraitb2)  # Discretised z-trait space. Each [traits] = 1
		if nTraity > 1:
			self.ydTrait = self.yTraits[1] - self.yTraits[0]  # Distance between y-traits. [dTrait] = 1
		else:
			self.ydTrait = np.nan
		if nTraitb2 > 1:
			self.b2dTrait = self.b2Traits[1] - self.b2Traits[0]  # Distance between z-traits. [dTrait] = 1
		else:
			self.b2dTrait = np.nan

		# State, including an initial state state0All which is a the state variable for the discretised  trait list
		self.state0 = state0  # Initial state variable used in steady state calculations. Each [state0] = [N]
		self.state0All = np.zeros((nTraity+1)*(nTraitb2+1))
		if nTraity > 1:
			self.y0ind = round((self.yInit-self.ytraitMin)/self.ydTrait)
		else:
			self.y0ind = 0
		if self.nTraitb2 > 1:
			self.b20ind = round((self.b2Init-self.b2traitMin)/self.b2dTrait)
		else:
			self.b20ind = 0
		self.state0All[0] = state0[0]
		self.state0All[1+self.y0ind] = state0[1]
		self.state0All[1+nTraity+self.b20ind] = state0[2]

		# Time
		self.ntEco = ntEco  # Number of ecological time-steps per evolutionary time-step. [ntEco] = 1
		self.TEco = TEco  # Final time of the ecological simulation. [TEco] = [T]
		self.dtEco = TEco/ntEco  # Ecological time-step. [dtEco] = [T]
		self.tEcoVec = np.linspace(0,TEco,ntEco)
		self.ntEvo = ntEvo  # Number of evolutionary time-steps. [ntEvo] = 1
		self.tEvoVec = np.arange(0,ntEvo+1)

	# Helper functions
	def birth(self, N):
		'Birth function'

		return(N*(self.a-self.q*N))

	def birthDer(self, N):
		'Derivative of the birth function'

		return(self.a-2*self.q*N)

	def birth2Der(self, N):
		'Second derivative of birth function'

		return(-2*self.q)

	def cost(self, x):
		'Cost function'

		if self.c2 != 0:
			return(self.c1*(1-np.exp(self.c2*x))/(1-np.exp(self.c2)))
		else:
			return(self.c1*x)

	def costDer(self, x):
		'Derivative of the cost function'

		if self.c2 != 0:
			return(-self.c1*self.c2*np.exp(self.c2*x)/(1-np.exp(self.c2)))
		else:
			return(self.c1)

	def cost2Der(self, x):
		'Second derivative of the cost function'

		if self.c2 != 0:
			return(-self.c1*self.c2**2*np.exp(self.c2*x)/(1-np.exp(self.c2)))
		else:
			return(0)

	# Transmission functions. Note we won't have one for the parasite as this is the trade-off parameter.
	def beta1fun(self, y):
		'Transmission for the mutualist. This is going to be a non-linear decrease using the cost function'

		return(self.beta1*(1-self.cost(y)))

	def beta1funDer(self, y):
		'Derivative of the transmission function'

		return(-self.beta1*self.costDer(y))

	def beta1fun2Der(self, y):
		'Second derivative of the transmission function'

		return(-self.beta1*self.cost2Der(y))

	# Virulence functions
	def alpha1fun(self, y):
		'Virulence of the mutualist as a funtion of its "effort" y. Use a constant function'

		return(self.alpha1)

	def alpha1funDer(self, y):
		'First derivative of the virulence of the mutualist with respect to the "effort" parameter y'

		return(0)

	def alpha1fun2Der(self, y):
		'Second derivative of the virulence of the mutualist with respect to the "effort" parameter y'

		return(0)

	def alpha2fun(self, b2):
		'Virulence of the parasite as a function of its transmission. We use a power law'
		
		return(self.alpha2*(1+b2**self.d))

	def alpha2funDer(self, b2):
		'First derivative of the virulence of the parasite with respect to the transmission'

		return(self.alpha2*self.d*(b2)**(self.d-1))

	def alpha2fun2Der(self, b2):
		'First derivative of the virulence of the parasite with respect to the transmission'
		
		return(self.alpha2*self.d*(self.d-1)*(b2)**(self.d-2))

	def alpha12fun(self, y, b2):
		'Coinfected virulence'

		return(self.alpha1fun(y)+self.alpha2fun(b2)*(1-(1-self.delta)*y))

	def alpha12funDery(self, y, b2):
		'First deriative of the coinfected virulence with respect to the mutualist "effort"'

		return(self.alpha1funDer(y)-self.alpha2fun(b2)*(1-self.delta))

	def alpha12funDerb2(self, y, b2):
		'First deriative of the coinfected virulence with respect to the parasite transmission'

		return(self.alpha2funDer(b2)*(1-(1-self.delta)*y))

	def alpha12fun2Deryy(self, y, b2):
		'Second deriative of the coinfected virulence with respect to tha parasite transmission'

		return(self.alpha1fun2Der(y))

	def alpha12fun2Deryb2(self, y, b2):
		'Cross derivative of the coinfected virulence'

		return(-self.alpha2funDer(b2)*(1-self.delta))

	def alpha12fun2Derb2b2(self, y, b2):
		'Second deriative of the coinfected virulence with respect to the parasite transmission'

		return(self.alpha2fun2Der(b2)*(1-(1-self.delta)*y))

	# Define a RHS function for the resident dynamics
	def resDynamics(self, t, state, yRes, b2Res):
		'Resident dynamics function'

		# Extract the initial values of S, I1, I2 and I12
		S = state[0]
		I1 = state[1]
		I2 = state[2]
		I12 = state[3]

		# Sum up to find N
		N = S + I1 + I2 + I12

		# Create the expressions for the RHS of the resident ODEs
		SRHS = self.birth(N) - (self.b + self.beta1fun(yRes)*(I1+I12) + b2Res*(I2+I12))*S + self.gamma1*I1 + self.gamma2*I2
		I1RHS = self.beta1fun(yRes)*S*(I1+I12) - (self.b + self.alpha1fun(yRes) + self.gamma1 + b2Res*(1-self.delta*yRes)*(I2+I12))*I1 + self.gamma2*I12
		I2RHS = b2Res*S*(I2+I12) - (self.b + self.alpha2fun(b2Res) + self.gamma2 + self.beta1fun(yRes)*(I1+I12))*I2 + self.gamma1*I12
		I12RHS = self.beta1fun(yRes)*I2*(I1+I12) + b2Res*(1-self.delta*yRes)*I1*(I2+I12) - (self.b + self.alpha12fun(yRes,b2Res) + self.gamma1 + self.gamma2)*I12

		return(np.array([SRHS, I1RHS, I2RHS, I12RHS]))

	# Create a function for finding the steady state of the resident dynamics
	def steadyState(self, yRes, b2Res, state0):
		'Function to evaluate the steady state of the resident dynamics'

		# Caluclate this with the numerical ODE solution
		stateODE = solve_ivp(lambda t, state: self.resDynamics(t, state, yRes, b2Res), (0,self.TEco), state0, method='BDF').y[:,-1]

		return(stateODE)

	def fitnessFunb2(self, b2Mut, yRes, b2Res):
		'''
		Fitness function for parasite
		'''

		# Steady state at resident trait values
		SS = self.steadyState(yRes, b2Res, self.state0)
		S = SS[0]
		I1 = SS[1]
		I2 = SS[2]
		I12 = SS[3]

		# Fitness function
		A = S*(self.b + self.gamma1 + self.gamma2 + self.alpha12fun(yRes, b2Mut) + self.beta1fun(yRes)*(I1+I12)) + I1*(1-self.delta*yRes)*(self.b + self.gamma1 + self.gamma2 + self.alpha2fun(b2Mut) + self.beta1fun(yRes)*(I1+I12))
		B = (self.b + self.gamma1 + self.gamma2 + self.alpha12fun(yRes, b2Mut))*(self.b + self.gamma2 + self.alpha2fun(b2Mut) + self.beta1fun(yRes)*(I1+I12)) - self.gamma1*self.beta1fun(yRes)*(I1+I12)

		return(b2Mut*A/B - 1)

	def selGrad(self, y, b2, t=0):
		'''
		Calculate the selection gradient at a position (y, b2) in trait space
		'''

		SS = self.steadyState(y, b2, self.state0)
		S = SS[0]
		I1 = SS[1]
		I2 = SS[2]
		I12 = SS[3]

		# Find the selection gradients at this value
		Am = S*(self.b + self.alpha12fun(y,b2) + self.gamma1 + self.gamma2 + b2*(1-self.delta*y)*(I2 + I12)) + I2*(self.b + self.alpha1fun(y) + self.gamma1 + self.gamma2 + b2*(1-self.delta*y)*(I2 + I12))
		Bm = (self.b + self.alpha12fun(y,b2) + self.gamma1 + self.gamma2)*(self.b + self.alpha1fun(y) + self.gamma1 + b2*(1-self.delta*y)*(I2 + I12)) - self.gamma2*b2*(1-self.delta*y)*(I2 + I12)
		derAm = S*(self.alpha12funDery(y,b2) - b2*self.delta*(I2 + I12)) + I2*(self.alpha1funDer(y) - b2*self.delta*(I2 + I12))
		derBm = self.alpha12funDery(y,b2)*(self.b + self.alpha1fun(y) + self.gamma1 + b2*(1-self.delta*y)*(I2 + I12)) + (self.alpha1funDer(y) - b2*self.delta*(I2 + I12))*(self.b + self.alpha12fun(y,b2) + self.gamma1 + self.gamma2) + self.gamma2*b2*self.delta*(I2 + I12)
		
		fitm = self.beta1fun(y)*Am/Bm-1
		selm = (self.beta1funDer(y)*Am + self.beta1fun(y)*derAm - (fitm + 1)*derBm)/Bm

		Ap = S*(self.b + self.alpha12fun(y,b2) + self.gamma1 + self.gamma2 + self.beta1fun(y)*(I1 + I12)) + I1*(1-self.delta*y)*(self.b + self.alpha2fun(b2) + self.gamma1 + self.gamma2 + self.beta1fun(y)*(I1 + I12))
		Bp = (self.b + self.alpha12fun(y,b2) + self.gamma1 + self.gamma2)*(self.b + self.alpha2fun(b2) + self.gamma2 + self.beta1fun(y)*(I1 + I12)) - self.gamma1*self.beta1fun(y)*(I1 + I12)
		derAp = S*self.alpha12funDerb2(y,b2) + I1*(1-self.delta*y)*self.alpha2funDer(b2)
		derBp = self.alpha12funDerb2(y,b2)*(self.b + self.alpha2fun(b2) + self.gamma2 + self.beta1fun(y)*(I1 + I12)) + self.alpha2funDer(b2)*(self.b + self.alpha12fun(y,b2) + self.gamma1 + self.gamma2)

		fitp = b2*Ap/Bp-1
		selp = (Ap + b2*derAp - (fitp + 1)*derBp)/Bp

		return(np.array([fitm, fitp, selm, selp]))

	# ODE RHS for the parasite
	def RHSPar(self, t, state, b2):
		'RHS for only the parasite present'

		# Extract S and I1
		S = state[0]
		I2 = state[1]

		# Calculate the S RHS
		SRHS = self.birth(S+I2) - (self.b + b2*I2)*S + self.gamma2*I2

		# Calcultae the I1 RHS
		I2RHS =b2*S*I2 - (self.b + self.gamma2 + self.alpha2fun(b2))*I2

		return(np.array([SRHS, I2RHS]))
		
	# Steady state for the parasite
	def SSPar(self, b2, state0):
		'Steady state when just the parasite is present'

		# ODE solution
		stateODE = solve_ivp(lambda t, state: self.RHSPar(t, state, b2), (0,self.TEco), state0).y[:,-1]

		return(stateODE)

	def plotPIPs(self, yRes=None, b2Res=None, n=101, showFig=False):
		'''
		Creates two PIPs, one for the mutualist evaluated at b2Res, and one for the parasite at yRes
		'''

		# Meshes
		if b2Res != None:
			yy = np.linspace(self.ytraitMin, self.ytraitMax, n)
			[YYr,YYm] = np.meshgrid(yy, yy)
		if yRes != None:
			bb = np.linspace(self.b2traitMin, self.b2traitMax, n)
			[BBr,BBm] = np.meshgrid(bb, bb)

		# Create storage matrices
		PIPMut = np.zeros((n, n))
		PIPPar = np.zeros((n, n))

		# Loop through and place fitness values in matrix
		for ii in range(n):
			for jj in range(n):
				if b2Res != None:
					PIPMut[ii,jj] = self.fitnessFuny(yy[ii], yy[jj], b2Res)
				if yRes != None:
					PIPPar[ii,jj] = self.fitnessFunb2(bb[ii], yRes, bb[jj])

		# Create a figure
		fig = plt.figure()
		
		# Plotting
		if b2Res != None and yRes != None:
			ax1 = fig.add_subplot(121)
			ax1.pcolormesh(yy, yy, PIPMut > 1e-6, cmap='Greys')
			ax1.set_xticks([0,0.5,1])
			ax1.set_xlabel('Resident')
			ax1.set_yticks([0,0.5,1])
			ax1.set_ylabel('Mutant')
			ax1.set_title('Mutualist')

			ax2 = fig.add_subplot(122)
			ax2.pcolormesh(bb, bb, PIPPar > 1e-6, cmap='Greys')
			ax2.set_xticks([0, self.b2traitMax/2, self.b2traitMax])
			ax2.set_xlabel('Resident')
			ax2.set_xlim(0,10)
			ax2.set_yticks([0, self.b2traitMax/2, self.b2traitMax])
			ax2.set_ylabel('Mutant')
			ax2.set_ylim(0,10)
			ax2.set_title('Parasite')

		elif b2Res != None:
			ax1 = fig.add_subplot(111)
			ax1.pcolormesh(yy, yy, PIPMut > 1e-6, cmap='Greys')
			ax1.set_xticks([0,0.5,1])
			ax1.set_xlabel('Resident')
			ax1.set_yticks([0,0.5,1])
			ax1.set_ylabel('Mutant')
		
		elif yRes != None:
			ax2 = fig.add_subplot(111)
			ax2.pcolormesh(bb, bb, PIPPar > 1e-6, cmap='Greys')
			ax2.set_xticks([0, self.b2traitMax/2, self.b2traitMax])
			ax2.set_xlabel('Resident')
			ax2.set_xlim(0, 10)
			ax2.set_yticks([0, self.b2traitMax/2, self.b2traitMax])
			ax2.set_ylabel('Mutant')
			ax2.set_xlim(0, 10)

		if showFig:
			plt.show()

		return(PIPMut, PIPPar)

	# Next we look at evolutionary dynamics
	def resDynamicsAll(self, t, state, indsMut=None, indsPar=None):
		'''
		Initial state is of size (nMut+1)*(nPar+1), where nMut is the number of non-zero 
		Mutualist trait values and nPar is the number of non-zero Parasite trait values.

		The state variable comes in in the order susceptible, mutualist traits, parasite traits, coevolved
		traits. Th coevolved traits run through the parasite trait first
		'''

		# If default (all) indices are needed, populate these
		if indsMut is None:
			indsMut = np.arange(self.nTraity)
		if indsPar is None:
			indsPar = np.arange(self.nTraitb2)

		# Calculate nMut and NPar
		nMut = len(indsMut)
		nPar = len(indsPar)

		# Extract each of the state variables
		S = state[0]
		I1vec = np.reshape(state[1:(1+nMut)], (nMut, ))
		I2vec = np.reshape(state[(1+nMut):(1+nMut+nPar)], (nPar,))
		I12mat = np.reshape(state[(1+nMut+nPar):], (nMut, nPar))
		I1tot = np.sum(I1vec)
		I2tot = np.sum(I2vec)
		I12tot = np.sum(I12mat)

		# Create the vectors in phenotypic space
		# yVec = np.reshape(self.yTraits[indsMut], (nMut, 1))
		# b2Vec = np.reshape(self.b2Traits[indsPar], (nPar, 1))
		YY = np.tile(self.yTraits[indsMut], (nPar, 1)).transpose()
		BB = np.tile(self.b2Traits[indsPar], (nMut,1))
		
		# Sum all to find N
		N = S + I1tot + I2tot + I12tot

		# Susceptibles
		SRHS = self.birth(N) - (self.b + np.sum(self.beta1fun(self.yTraits[indsMut])*(I1vec+np.sum(I12mat,1))) + np.sum(self.b2Traits[indsPar]*(I2vec+np.sum(I12mat,0))))*S + self.gamma1*I1tot + self.gamma2*I2tot

		# Infected with the mutualist
		I1RHS = self.beta1fun(self.yTraits[indsMut])*S*(I1vec+np.sum(I12mat,1)) - (self.b + self.gamma1*I1vec + self.alpha1fun(self.yTraits) + (1-self.delta*self.yTraits[indsMut])*np.sum(self.b2Traits[indsPar]*(I2vec+np.sum(I12mat, 0))))*I1vec + self.gamma2*np.sum(I12mat,1)

		# Infected with the parasite
		I2RHS = self.b2Traits[indsPar]*S*(I2vec+np.sum(I12mat,0)) - (self.b + self.gamma2*I2vec + self.alpha2fun(self.b2Traits[indsPar]) + np.sum(self.beta1fun(self.yTraits[indsMut])*(I1vec+np.sum(I12mat,1))))*I2vec + self.gamma1*np.sum(I12mat,0)

		# Coinfected
		I12RHS = (np.tile(I2vec,(nMut,1))*np.tile(self.beta1fun(self.yTraits[indsMut])*(I1vec + np.sum(I12mat,1)),(nPar,1)).transpose() + np.tile((1-self.delta*self.yTraits[indsMut])*I1vec,(nPar,1)).transpose()*np.tile(self.b2Traits[indsPar]*(I2vec+np.sum(I12mat,0)), (nMut,1)) - (self.b + self.gamma1 + self.gamma2 + self.alpha12fun(YY,BB))*I12mat)

		return(np.hstack((SRHS, I1RHS, I2RHS, np.reshape(I12RHS,(nMut*nPar,)))))

	# Create a function for finding the steady state of the resident dynamics for all parasite traits
	def steadyStateAll(self, state0, indsMut=None, indsPar = None):
		'Steady state of the discretised system with initial condition state0, mutualist trait value of yRes, and where only the indices in inds are non-zero. If inds is not specified, solve the full system'

		if indsMut is None:
			indsMut = np.arange(self.nTraity)
		if indsPar is None:
			indsPar = np.arange(self.nTraitb2)

		# Calculate nMut and NPar
		nMut = len(indsMut)
		nPar = len(indsPar)

		# Find the elements of the coevolved state these indices correspond to by converting to a matrix, selecting indices, and converting back
		I12mat = np.reshape(state0[(1+self.nTraity+self.nTraitb2):], (self.nTraity,self.nTraitb2))
		I12mat = I12mat[indsMut,][:,indsPar]

		# Calculate the new initial condition
		state0New = np.hstack((state0[0], state0[1+indsMut], state0[1+self.nTraity+indsPar], np.reshape(I12mat, (nMut*nPar,))))

		stateODE = solve_ivp(lambda t, state: self.resDynamicsAll(t, state, indsMut, indsPar), (0,self.TEco), y0=state0New, method='Radau').y[:,-1]

		return(stateODE)

	# Function for the ecolutionary simulation
	def evoSim(self, directory='', saveData=False):

		# Define the filename
		filename = str(directory + datetime.now().strftime("%y%m%d%H%M%S"))

		# Find the initial indices
		yIndInit = self.y0ind
		b2IndInit = self.b20ind

		# Create a variable which is a 1 if the trait is evolving, and zero otherwise
		y0evo = 1*(self.nTraity > 1) + 0
		b20evo = 1*(self.nTraitb2 > 1) + 0

		# Store trait lists and matrices for each trait
		yTraitListMat = np.zeros((self.ntEvo+1, self.nTraity))
		yTraitList = np.zeros((self.nTraity))
		yTraitList[yIndInit] = 1
		yTraitListMat[0,] = yTraitList

		b2TraitListMat = np.zeros((self.ntEvo+1, self.nTraitb2))
		b2TraitList = np.zeros((self.nTraitb2))
		b2TraitList[b2IndInit] = 1
		b2TraitListMat[0,] = b2TraitList

		# Initialise the storage matrix
		store = np.zeros((self.ntEvo+1, (self.nTraity+1)*(self.nTraitb2+1)))
		state = self.state0All
		store[0,] = state

		# Loop through evolutionary time
		for nn in range(self.ntEvo):

			# Write the state before to a variable
			stateBef = state

			# Locate all nonzero indices
			indsMut = yTraitList*(1+np.arange(self.nTraity))
			indsMut = np.array(indsMut[indsMut>0], dtype=int)-1
			nMut = len(indsMut)

			indsPar = b2TraitList*(1+np.arange(self.nTraitb2))
			indsPar = np.array(indsPar[indsPar>0], dtype=int)-1
			nPar = len(indsPar)

			# Find the steady state and extract each part
			stateAfter = self.steadyStateAll(stateBef, indsMut=indsMut, indsPar=indsPar)
			S = stateAfter[0]
			I1 = stateAfter[1:(1+nMut)]
			I2 = stateAfter[(1+nMut):(1+nMut+nPar)]
			I12 = np.reshape(stateAfter[(1+nMut+nPar):], (nMut,nPar))

			# Calculate the frequencies for each y and b2 trait value
			yFreqs = I1 + np.sum(I12, axis=1)
			b2Freqs = I2 + np.sum(I12, axis=0)

			# Find the indices where the extinction threshold has been met
			yExtinct = (yFreqs < self.extinct)*(1+np.arange(nMut))
			yExtinct = yExtinct[yExtinct>0]-1

			b2Extinct = (b2Freqs < self.extinct)*(1+np.arange(nPar))
			b2Extinct = b2Extinct[b2Extinct>0]-1

			# Set anything below the threshold to be 0 or delete them from the temporary storage arrays
			I1 = np.delete(I1, yExtinct)
			I2 = np.delete(I2, b2Extinct)
			I12 = np.delete(I12, yExtinct, axis=0)
			I12 = np.delete(I12, b2Extinct, axis=1)
			yTraitList[indsMut[yExtinct]] = 0
			b2TraitList[indsPar[b2Extinct]] = 0
			yFreqs = np.delete(yFreqs, yExtinct)
			b2Freqs = np.delete(b2Freqs, b2Extinct)
			indsMut = np.delete(indsMut, yExtinct)
			indsPar = np.delete(indsPar, b2Extinct)
			nMut = len(indsMut)
			nPar = len(indsPar)

			# Reconstruct the full state variable for all but the coevolved part
			state = np.zeros((self.nTraity+1)*(self.nTraitb2+1))
			state[0] = S
			state[1+indsMut] = I1
			state[1+self.nTraity+indsPar] = I2

			# Create a full sized coinfected matrix, place the coinfected states into the appropriate places and reshape
			coinfected = np.zeros((self.nTraity, self.nTraitb2))
			for rowInd in range(nMut):
				fullRowInd = indsMut[rowInd]
				for colInd in range(nPar):
					fullColInd = indsPar[colInd]
					coinfected[fullRowInd, fullColInd] = I12[rowInd, colInd]
			coinfectedReshaped = np.reshape(coinfected, (self.nTraity*self.nTraitb2,))
			state[(1+self.nTraity+self.nTraitb2):] = coinfectedReshaped

			# Now we mutate. We will choose one of the two trait values with probability proportional to the frequencies of each type
			sFreqs = sum(yFreqs) + sum(b2Freqs)
			rFreqs = sFreqs*np.random.rand()

			# Choose a random number
			randNum = np.random.rand()

			# Mutualist trait
			if randNum < y0evo/(y0evo+b20evo):
				if sum(yTraitList) > 0:
					yFreqsSum = np.sum(yFreqs)
					yFreqsRand = np.random.rand()*yFreqsSum
					yFreqsCum = yFreqs[0]
					yInd = 0
					while yFreqsCum < yFreqsRand:
						yInd += 1
						yFreqsCum += yFreqs[yInd]
					if indsMut[yInd] == 0:
						yIndNew = 1
					elif indsMut[yInd] == self.nTraity-1:
						yIndNew = self.nTraity-2
					else:
						yIndNew = round(indsMut[yInd] + np.sign(np.random.rand()-0.5))
					state[1+yIndNew] += 1e-3
					yTraitList[yIndNew] = 1
				yTraitListMat[nn+1,:] = yTraitList
				b2TraitListMat[nn+1,:] = b2TraitList

			# Parasite trait
			elif randNum < 1:
				if sum(b2TraitList) > 0:
					b2FreqsSum = np.sum(b2Freqs)
					b2FreqsRand = np.random.rand()*b2FreqsSum
					b2FreqsCum = b2Freqs[0]
					b2Ind = 0
					while b2FreqsCum < b2FreqsRand:
						b2Ind += 1
						b2FreqsCum += b2Freqs[b2Ind]
					if indsPar[b2Ind] == 0:
						b2IndNew = 1
					elif indsPar[b2Ind] == self.nTraitb2-1:
						b2IndNew = self.nTraitb2-2
					else:
						b2IndNew = round(indsPar[b2Ind] + np.sign(np.random.rand()-0.5))
					state[1+self.nTraity+b2IndNew] += 1e-3
					b2TraitList[b2IndNew] = 1
				yTraitListMat[nn+1,:] = yTraitList
				b2TraitListMat[nn+1,:] = b2TraitList

			# Store the state variable
			store[nn+1,:] = state

		if saveData == True:

			# Check if the save directory exists. If not, create it
			isDirSave = os.path.isdir(directory)
			if not isDirSave:
				os.mkdir(directory)

			# Now we save the workspace
			pdict = locals()
			file = open('%s.pkl' % filename, 'wb')
			pickle.dump(pdict, file)
			file.close()

			# Add the parameters to the dataset dataframe
			# Create the dictionary
			oldDict = vars(self)
			newDict = {key: val for key, val in oldDict.items() if not isinstance(val, np.ndarray)}
			newDict['filename'] = filename

			# Append to the dataframe
			filepath = Path('Datasets/datasets.csv')
			filepath.parent.mkdir(parents=True, exist_ok=True)
			out = pd.read_csv(filepath)
			df2 = pd.DataFrame(newDict, index=['EvoSim'])
			out = pd.concat([out,df2])
			out.to_csv(filepath, index=False)

	def edgeStep(self, rect, n):
		'''
		Calculates where the dynamics should tend to if the zero is outside the region of interest
		'''

		# Extract the elements of rect
		a0 = rect[0]
		b0 = rect[1]
		c0 = rect[2]
		d0 = rect[3]

		# Create meshes
		yMesh = np.linspace(a0, b0, n)
		b2Mesh = np.linspace(c0, d0, n)

		# Find the selection gradient along each edge in the approporiate selection gradient.
		# We look for zeros on the b2 boundary for the mutualist's SG, and zeros on the y boundaries for the parasite's SG
		yLowBoundary = np.zeros(n)
		yUpBoundary = np.zeros(n)
		b2LowBoundary = np.zeros(n)
		b2UpBoundary = np.zeros(n)
		for ii in range(n):
			yLowBoundary[ii] = self.selGrad(a0, b2Mesh[ii])[3]
			yUpBoundary[ii] = self.selGrad(b0, b2Mesh[ii])[3]
			b2LowBoundary[ii] = self.selGrad(yMesh[ii], c0)[2]
			b2UpBoundary[ii] = self.selGrad(yMesh[ii], d0)[2]			

		# Find indices jj such that a zero appears between jj and jj+1
		yLowInds = (yLowBoundary[:-1]*yLowBoundary[1:] < 0)*np.arange(1,n)
		yLowInds = yLowInds[yLowInds>0] - 1
		nyLow = len(yLowInds)
		yUpInds = (yUpBoundary[:-1]*yUpBoundary[1:] < 0)*np.arange(1,n)
		yUpInds = yUpInds[yUpInds>0] - 1
		nyUp = len(yUpInds)
		b2LowInds = (b2LowBoundary[:-1]*b2LowBoundary[1:] < 0)*np.arange(1,n)
		b2LowInds = b2LowInds[b2LowInds>0] - 1
		nb2Low = len(b2LowInds)
		b2UpInds = (b2UpBoundary[:-1]*b2UpBoundary[1:] < 0)*np.arange(1,n)
		b2UpInds = b2UpInds[b2UpInds>0] - 1
		nb2Up = len(b2UpInds)

		# Find the zeros using interpolation and add its singular strategy
		edgeCaseZeros = np.zeros((nyLow+nyUp+nb2Low+nb2Up,3))

		# Lower y boundary
		for jj in range(nyLow):
			b20 = b2Mesh[yLowInds[jj]]
			b21 = b2Mesh[yLowInds[jj]+1]
			f0 = yLowBoundary[yLowInds[jj]]
			f1 = yLowBoundary[yLowInds[jj]+1]
			edgeCaseZeros[jj,0] = a0
			edgeCaseZeros[jj,1] = b20 - f0/(f1-f0)*(b21-b20)
			edgeCaseZeros[jj,2] = self.selGrad(edgeCaseZeros[jj,0], edgeCaseZeros[jj,1])[3]

		# Upper y boundary
		for jj in range(nyUp):
			b20 = b2Mesh[yUpInds[jj]]
			b21 = b2Mesh[yUpInds[jj]+1]
			f0 = yUpBoundary[yUpInds[jj]]
			f1 = yUpBoundary[yUpInds[jj]+1]
			edgeCaseZeros[jj+nyLow,0] = b0
			edgeCaseZeros[jj+nyLow,1] = b20 - f0/(f1-f0)*(b21-b20)
			edgeCaseZeros[jj+nyLow,2] = self.selGrad(edgeCaseZeros[jj+nyLow,0], edgeCaseZeros[jj+nyLow,1])[3]

		# Lower b2 boundary
		for jj in range(nb2Low):
			y0 = yMesh[b2LowInds[jj]]
			y1 = yMesh[b2LowInds[jj]+1]
			f0 = b2LowBoundary[b2LowInds[jj]]
			f1 = b2LowBoundary[b2LowInds[jj]+1]
			edgeCaseZeros[jj+nyLow+nyUp,0] = y0 - f0/(f1-f0)*(y1-y0)
			edgeCaseZeros[jj+nyLow+nyUp,1] = c0
			edgeCaseZeros[jj+nyLow+nyUp,2] = self.selGrad(edgeCaseZeros[jj+nyLow+nyUp,0], edgeCaseZeros[jj+nyLow+nyUp,1])[2]

		# Upper b2 boundary
		for jj in range(nb2Up):
			y0 = yMesh[b2UpInds[jj]]
			y1 = yMesh[b2UpInds[jj]+1]
			f0 = b2UpBoundary[b2UpInds[jj]]
			f1 = b2UpBoundary[b2UpInds[jj]+1]
			edgeCaseZeros[jj+nyLow+nyUp+nb2Low,0] = y0 - f0/(f1-f0)*(y1-y0)
			edgeCaseZeros[jj+nyLow+nyUp+nb2Low,1] = d0
			edgeCaseZeros[jj+nyLow+nyUp+nb2Low,2] = self.selGrad(edgeCaseZeros[jj+nyLow+nyUp+nb2Low,0], edgeCaseZeros[jj+nyLow+nyUp+nb2Low,1])[2]

		return(edgeCaseZeros)

# Create functions for plotting which will be used to create gifs and figures for the paper
def plotEvoandEco(filename, directoryToSave, index=None, c1=False, c2=False, outData=False):
	'''
	Plots an evolutionary simulation with the data (output from Simulation.evoSim) provided in filename.pkl.
	Also plots the steady states for the four states
	NOTE: Do not put the .pkl extension in your filename
	'''

	# Create an index if none is specified
	if index == None:
		index = 'TEST'

	# Load the data
	file = open('%s.pkl' % filename, 'rb')
	pdict = pickle.load(file)
	file.close()

	# Set any required data to a variable
	store = pdict['store']
	self = pdict['self']

	# Extract the required plotting data
	# Ecological data
	S = store[:,0]
	I1 = np.sum(store[:,1:(1+self.nTraity)], axis=1)
	I2 = np.sum(store[:,(1+self.nTraity):(1+self.nTraity+self.nTraitb2)], axis=1)
	I12 = np.sum(store[:,(1+self.nTraity+self.nTraitb2):], axis=1)

	# Frequencies for each variant
	IIy = store[:,1:(1+self.nTraity)]
	IIb2 = store[:,(1+self.nTraity):(1+self.nTraity+self.nTraitb2)]
	for t in range(self.ntEvo+1):
		I12mat = np.reshape(store[t, (1+self.nTraity+self.nTraitb2):], (self.nTraity, self.nTraitb2))
		IIy[t,] += np.sum(I12mat, axis=1)
		IIb2[t,] += np.sum(I12mat, axis=0)

	# Create some colormaps for the evolutionary plots
	cmapMut = ListedColormap(['w', '#1f78b4'])
	cmapPar = ListedColormap(['w', '#33a02c'])

	# Set up a figure
	plotWidth = 11.69
	plotHeight = 8.27
	fig = plt.figure(figsize=(plotWidth, plotHeight))
	gs = fig.add_gridspec(2,4)
	if c1:
		fig.suptitle(r'$c_1 = $' + ('%.2f' % self.c1))
	elif c2:
		fig.suptitle(r'$c_2 = $' + ('%.2f' % self.c2))

	# We want three figures, two tall ones with evolutionary simulations
	# and a third one with the ecological dynamics
	# Evo plot for mutualist
	ax1 = fig.add_subplot(gs[:,0])
	ax1.pcolormesh(self.yTraits, np.arange(self.ntEvo+1), IIy > 1e-3, cmap=cmapMut, vmin=0, vmax=1)
	ax1.set_xlabel(r'$y$')
	ax1.set_xticks([0,0.5,1])
	ax1.set_ylabel('Evolutionary Time')
	ax1.set_yticks([0,self.ntEvo/2,self.ntEvo])

	# Evo plot for parasite
	ax2 = fig.add_subplot(gs[:,-1])
	ax2.pcolormesh(self.b2Traits, np.arange(self.ntEvo+1), IIb2 > 1e-3, cmap=cmapPar, vmin=0, vmax=1)
	ax2.set_xlabel(r'$\beta_2$')
	ax2.set_xticks([0,self.b2traitMax/2,self.b2traitMax])
	ax2.set_ylabel('Evolutionary Time')
	ax2.set_yticks([0,self.ntEvo/2,self.ntEvo])

	# Plot for eco dynamics
	ax3 = fig.add_subplot(gs[:,1:-1])
	ax3.plot(np.arange(1,self.ntEvo+1), S[1:], color='#a6cee3', lw=2, label='Sus')
	ax3.plot(np.arange(1,self.ntEvo+1), I1[1:], color='#1f78b4', lw=2, label='Mut')
	ax3.plot(np.arange(1,self.ntEvo+1), I2[1:], color='#33a02c', lw=2, label='Par')
	ax3.plot(np.arange(1,self.ntEvo+1), I12[1:], color='#b2df8a', lw=2, label='Coinf')
	ax3.set_xlabel('Evolutionary Time')
	ax3.set_xticks([0,self.ntEvo/2,self.ntEvo])
	ax3.set_ylabel('Density')
	ax3.set_ylim(0,2)
	ax3.legend(loc='upper right')
	
	plt.tight_layout()

	if outData:
		return([IIy, IIb2])
	fig.clf()
	plt.close()

# Function for running over data and saving figures
def plotOverParams(directoryToLoad, directoryToSave, c1=False, c2=False):
	'''
	Code to save figures over a parameter run, with data stored in directoryToLoad
	'''

	# Check the load directory exists, if not exist with a message
	isDirLoad = os.path.isdir(directoryToLoad)
	if not isDirLoad:
		print('No such directory containing data')
		return

	# Check if the save directory exists, if not create one
	isDirSave = os.path.isdir(directoryToSave)
	if not isDirSave:
		os.mkdir(directoryToSave)

	# Loop through all files in the directory and make a plot.
	# Name the plot with an index from 0 to number of plots
	filesInDir = [f for f in os.listdir(directoryToLoad) if os.path.isfile(os.path.join(directoryToLoad, f))]
	nFiles = len(filesInDir)

	# Check if the number of files is 1. If so, set the outData Boolean to true so that it outputs the data
	outData = nFiles == 1

	# Loop over files
	for ii in range(nFiles):

		# Extract the filename
		filename = filesInDir[ii][:-4]

		# Plot
		data = plotEvoandEco(directoryToLoad + filename, directoryToSave, index=ii, c1=c1, c2=c2, outData=outData)

	return(data)

# Functions for creating a heat map which will demonstrate classification
def heatmapData(directoryToSave, yInit=0, n=101, delta=0):
	'''
	This function will generate the data required in order to create a classification heatmap
	'''

	# Check if the save directory exists, if not create one
	isDirSave = os.path.isdir(directoryToSave)
	if not isDirSave:
		os.mkdir(directoryToSave)

	# Create vectors for c1 and c2
	c1Vec = np.linspace(0, 1, n)
	c2Vec = np.linspace(-5, 5, n)

	# Loop through each, run an evolutionary simulation, and save data
	for ii in range(n):
		for jj in range(n):
			sim = Simulation(nTraity=21, nTraitb2=21, c1=c1Vec[ii], c2=c2Vec[jj], ntEvo=500, yInit=yInit, delta=delta)
			sim.evoSim(directoryToSave, saveData=True)

def initialDataframeClassify(dir0, dir1, saveDir, savename):
	'''
	Looks in two directories which contain the initial coarse grained datasets
	dir0 contains the files for yInit=0
	dir1 contains the files for yInit=1
	Outputs a dataframe which is also saved into a directory saveDir
	'''

	# Create an empty dataframe
	df = pd.DataFrame(columns=['level', 'c1', 'c2', 'y0Val', 'y1Val', 'class'])

	# Find the number of datasets in each directory
	filesInDir0 = [f for f in os.listdir(dir0) if os.path.isfile(os.path.join(dir0, f))]
	nFiles0 = len(filesInDir0)
	nFilesPerDir0 = np.sqrt(nFiles0).astype('int')
	filesInDir1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
	nFiles1 = len(filesInDir1)
	nFilesPerDir1 = np.sqrt(nFiles1).astype('int')

	# Check the values are the same
	if nFilesPerDir0 != nFilesPerDir1:
		print('Different number of datasets per directory')
		return

	# Loop through the datasets. This will only work if the directories are in the same order
	for ii in range(nFiles0):

		# Extract the filename
		filename0 = filesInDir0[ii][:-4]
		filename1 = filesInDir1[ii][:-4]

		# Load the data
		file0 = open('%s%s.pkl' % (dir0,filename0), 'rb')
		pdict0 = pickle.load(file0)
		file0.close()
		file1 = open('%s%s.pkl' % (dir1,filename1), 'rb')
		pdict1 = pickle.load(file1)
		file1.close()

		# Find the final value of y in each dataset
		self0 = pdict0["self"]
		self1 = pdict1["self"]
		yTraits0 = pdict0["yTraitListMat"][-1,:]
		yTraits1 = pdict1["yTraitListMat"][-1,:]
		store0 = pdict0['store']
		store1 = pdict1['store']
		storeFinaly0 = store0[-1, 1:(1+self0.nTraity)] + np.sum(np.reshape(store0[-1, (1+self0.nTraity+self0.nTraitb2):], (self0.nTraity, self0.nTraitb2)), axis=1)
		storeFinaly1 = store1[-1, 1:(1+self1.nTraity)] + np.sum(np.reshape(store1[-1, (1+self1.nTraity+self1.nTraitb2):], (self1.nTraity, self1.nTraitb2)), axis=1)
		if np.sum(yTraits0) > 0:
			averageValy0 = np.sum(yTraits0*self0.yTraits*storeFinaly0)/sum(storeFinaly0)
		else:
			averageValy0 = np.nan
		if np.sum(yTraits1) > 0:
			averageValy1 = np.sum(yTraits1*self1.yTraits*storeFinaly1)/sum(storeFinaly1)
		else:
			averageValy1 = np.nan

		# Series of Boolean variables to help classify
		max0 = np.abs(averageValy0-1) < 1e-3
		min0 = np.abs(averageValy0) < 1e-3
		ext0 = np.isnan(averageValy0)
		max1 = np.abs(averageValy1-1) < 1e-3
		min1 = np.abs(averageValy1) < 1e-3
		ext1 = np.isnan(averageValy1)

		# Classify
		if max0 and max1 and not ext0 and not ext1:
			classifyVal = 1
		if min0 and min1 and not ext0 and not ext1:
			classifyVal = 2
		if (max0 and min1) or (min0 and max1) and not ext0 and not ext1:
			classifyVal = 3
		if (not max0 and not min0) and (not max0 and not min0) and not ext0 and not ext0:
			classifyVal = 4
		if ((max0 and (not max1 and not min1)) or ((not max0 and not min0) and max1)) and not ext0 and not ext1:
			classifyVal = 1 #5
		if ((min0 and (not max1 and not min1)) or ((not max0 and not min0) and min1)) and not ext0 and not ext1:
			classifyVal = 2 #6
		if (min0 and ext1) or (min1 and ext0):
			classifyVal = 2 #7 
		if ((not max0 and not min0) and ext1) or (ext0 and (not max1 and not min1)):
			classifyVal = 4 #8

		# Create a dictionary to append to the dataframe
		newDict = {'level': 0, 'c1': self0.c1, 'c2': self0.c2, 'y0Val': averageValy0, 'y1Val': averageValy1, 'class': classifyVal}

		# Append to the dataframe
		dfNew = pd.DataFrame(newDict, index=[ii])
		df = pd.concat([df, dfNew], ignore_index = True)

	# Check if the save directory exists, if not create one
	isDirSave = os.path.isdir(saveDir)
	if not isDirSave:
		os.mkdir(saveDir)

	# Save the dataset
	df.to_csv(saveDir + savename, index=False)

	return(df)

def refinementStep(dataFrame, level, v1, v2, refDir0, refDir1, delta=0):
	'''
	Takes a dataframe and either imputes data or generates new datasets to classify
	dataFrame is the orginal dataFrame
	level is the refinement level
	v1, v2 are the vector of values for the parameter sweep
	refDir0, refDir1 are the directories to place refinement data in
	'''

	# First we need to reconstruct the matrx
	# Load the dataFrame
	df = pd.read_csv(dataFrame)

	# Check that the two directories exist. If they don't, create them
	isRefDir0 = os.path.isdir(refDir0)
	if not isRefDir0:
		os.mkdir(refDir0)
	isRefDir1 = os.path.isdir(refDir1)
	if not isRefDir1:
		os.mkdir(refDir1)

	# Check that the number of elements in the dataframe is the same as the product of the number of elements in the vectors
	ndf = len(df)
	nv1 = len(v1)
	nv2 = len(v2)
	if ndf != nv1*nv2:
		return('Error: dataframe length does not math the vectors v1 and v2')

	# Calculate the vector differences (assumes regular mesh)
	dv1 = v1[1] - v1[0]
	dv2 = v2[1] - v2[0]
	
	# Reconstruct the matrix
	M = np.zeros((nv2, nv1))
	for ii in range(ndf):
		M[round((df['c2'][ii]+5)/dv2), round(df['c1'][ii]/dv1)] = df['class'][ii]

	# Create parameter vectors and a matrix which are twice as big as the other
	v1New = np.arange(v1[0], v1[-1]+dv1/2, dv1/2)
	nv1New = len(v1New)
	v2New = np.arange(v2[0], v2[-1]+dv1/2, dv2/2)
	nv2New = len(v2New)
	MNew = np.nan*np.ones((nv2New, nv1New))
	for ii in range(nv1New):
		for jj in range(nv2New):
			MNew[jj,ii] = M[round(jj/2), round(ii/2)]*(ii % 2 == 0 and jj % 2 == 0)

	# Run along the even rows and impute as necessary
	for ii in range(1, nv1New, 2):
		for jj in range(0, nv2New, 2):
			left = MNew[jj, ii-1]
			right = MNew[jj, ii+1]
			if left == right:
				MNew[jj, ii] = MNew[jj, ii-1]

				# Create a dictionary to append to the dataframe
				newDict = {'level': level, 'c1': v1New[ii], 'c2': v2New[jj], 'y0Val': np.nan, 'y1Val': np.nan, 'class': MNew[jj, ii]}

				# Append to the dataframe
				dfNew = pd.DataFrame(newDict, index=[ii])
				df = pd.concat([df, dfNew], ignore_index = True)
			else:
				# Create new datasets
				sim0 = Simulation(nTraity=21, nTraitb2=21, c1=v1New[ii], c2=v2New[jj], ntEvo=500, yInit=0, delta=delta)
				sim1 = Simulation(nTraity=21, nTraitb2=21, c1=v1New[ii], c2=v2New[jj], ntEvo=500, yInit=1, delta=delta)
				sim0.evoSim(refDir0, True)
				sim1.evoSim(refDir1, True)

				# Extract the two files that we have just created
				createdFile0 = [f for f in os.listdir(refDir0) if os.path.isfile(os.path.join(refDir0, f))][-1][:-4]
				createdFile1 = [f for f in os.listdir(refDir1) if os.path.isfile(os.path.join(refDir1, f))][-1][:-4]
				
				# Load the data
				file0 = open('%s%s.pkl' % (refDir0,createdFile0), 'rb')
				pdict0 = pickle.load(file0)
				file0.close()
				file1 = open('%s%s.pkl' % (refDir1,createdFile1), 'rb')
				pdict1 = pickle.load(file1)
				file1.close()

				# Find the final value of y in each dataset
				self0 = pdict0["self"]
				self1 = pdict1["self"]
				yTraits0 = pdict0["yTraitListMat"][-1,:]
				yTraits1 = pdict1["yTraitListMat"][-1,:]
				store0 = pdict0['store']
				store1 = pdict1['store']
				storeFinaly0 = store0[-1, 1:(1+self0.nTraity)] + np.sum(np.reshape(store0[-1, (1+self0.nTraity+self0.nTraitb2):], (self0.nTraity, self0.nTraitb2)), axis=1)
				storeFinaly1 = store1[-1, 1:(1+self1.nTraity)] + np.sum(np.reshape(store1[-1, (1+self1.nTraity+self1.nTraitb2):], (self1.nTraity, self1.nTraitb2)), axis=1)
				if np.sum(yTraits0) > 0:
					averageValy0 = np.sum(yTraits0*self0.yTraits*storeFinaly0)/sum(storeFinaly0)
				else:
					averageValy0 = np.nan
				if np.sum(yTraits1) > 0:
					averageValy1 = np.sum(yTraits1*self1.yTraits*storeFinaly1)/sum(storeFinaly1)
				else:
					averageValy1 = np.nan

				# Series of Boolean variables to help classify
				max0 = np.abs(averageValy0-1) < 1e-3
				min0 = np.abs(averageValy0) < 1e-3
				ext0 = np.isnan(averageValy0)
				max1 = np.abs(averageValy1-1) < 1e-3
				min1 = np.abs(averageValy1) < 1e-3
				ext1 = np.isnan(averageValy1)

				# Classify
				if max0 and max1 and not ext0 and not ext1:
					classifyVal = 1
				if min0 and min1 and not ext0 and not ext1:
					classifyVal = 2
				if (max0 and min1) or (min0 and max1) and not ext0 and not ext1:
					classifyVal = 3
				if (not max0 and not min0) and (not max0 and not min0) and not ext0 and not ext0:
					classifyVal = 4
				if ((max0 and (not max1 and not min1)) or ((not max0 and not min0) and max1)) and not ext0 and not ext1:
					classifyVal = 1 #5
				if ((min0 and (not max1 and not min1)) or ((not max0 and not min0) and min1)) and not ext0 and not ext1:
					classifyVal = 2 #6
				if (min0 and ext1) or (min1 and ext0):
					classifyVal = 2 #7 
				if ((not max0 and not min0) and ext1) or (ext0 and (not max1 and not min1)):
					classifyVal = 4 #8

				# Create a dictionary to append to the dataframe
				newDict = {'level': level, 'c1': self0.c1, 'c2': self0.c2, 'y0Val': averageValy0, 'y1Val': averageValy1, 'class': classifyVal}

				# Append to the dataframe
				dfNew = pd.DataFrame(newDict, index=[ii])
				df = pd.concat([df, dfNew], ignore_index = True)

				# Add to matrix
				MNew[jj, ii] = classifyVal

	# Run along the even columns and impute as necessary
	for ii in range(0, nv1New, 2):
		for jj in range(1, nv2New, 2):
			down = MNew[jj-1, ii]
			up = MNew[jj+1, ii]
			if down == up:
				MNew[jj, ii] = MNew[jj-1, ii]
				
				# Create a dictionary to append to the dataframe
				newDict = {'level': level, 'c1': v1New[ii], 'c2': v2New[jj], 'y0Val': np.nan, 'y1Val': np.nan, 'class': MNew[jj, ii]}

				# Append to the dataframe
				dfNew = pd.DataFrame(newDict, index=[ii])
				df = pd.concat([df, dfNew], ignore_index = True)
			else:
				# Create new datasets
				sim0 = Simulation(nTraity=21, nTraitb2=21, c1=v1New[ii], c2=v2New[jj], ntEvo=500, yInit=0, delta=delta)
				sim1 = Simulation(nTraity=21, nTraitb2=21, c1=v1New[ii], c2=v2New[jj], ntEvo=500, yInit=1, delta=delta)
				sim0.evoSim(refDir0, True)
				sim1.evoSim(refDir1, True)

				# Extract the two files that we have just created
				createdFile0 = [f for f in os.listdir(refDir0) if os.path.isfile(os.path.join(refDir0, f))][-1][:-4]
				createdFile1 = [f for f in os.listdir(refDir1) if os.path.isfile(os.path.join(refDir1, f))][-1][:-4]
				
				# Load the data
				file0 = open('%s%s.pkl' % (refDir0,createdFile0), 'rb')
				pdict0 = pickle.load(file0)
				file0.close()
				file1 = open('%s%s.pkl' % (refDir1,createdFile1), 'rb')
				pdict1 = pickle.load(file1)
				file1.close()

				# Find the final value of y in each dataset
				self0 = pdict0["self"]
				self1 = pdict1["self"]
				yTraits0 = pdict0["yTraitListMat"][-1,:]
				yTraits1 = pdict1["yTraitListMat"][-1,:]
				store0 = pdict0['store']
				store1 = pdict1['store']
				storeFinaly0 = store0[-1, 1:(1+self0.nTraity)] + np.sum(np.reshape(store0[-1, (1+self0.nTraity+self0.nTraitb2):], (self0.nTraity, self0.nTraitb2)), axis=1)
				storeFinaly1 = store1[-1, 1:(1+self1.nTraity)] + np.sum(np.reshape(store1[-1, (1+self1.nTraity+self1.nTraitb2):], (self1.nTraity, self1.nTraitb2)), axis=1)
				if np.sum(yTraits0) > 0:
					averageValy0 = np.sum(yTraits0*self0.yTraits*storeFinaly0)/sum(storeFinaly0)
				else:
					averageValy0 = np.nan
				if np.sum(yTraits1) > 0:
					averageValy1 = np.sum(yTraits1*self1.yTraits*storeFinaly1)/sum(storeFinaly1)
				else:
					averageValy1 = np.nan

				# Series of Boolean variables to help classify
				max0 = np.abs(averageValy0-1) < 1e-3
				min0 = np.abs(averageValy0) < 1e-3
				ext0 = np.isnan(averageValy0)
				max1 = np.abs(averageValy1-1) < 1e-3
				min1 = np.abs(averageValy1) < 1e-3
				ext1 = np.isnan(averageValy1)

				# Classify
				if max0 and max1 and not ext0 and not ext1:
					classifyVal = 1
				if min0 and min1 and not ext0 and not ext1:
					classifyVal = 2
				if (max0 and min1) or (min0 and max1) and not ext0 and not ext1:
					classifyVal = 3
				if (not max0 and not min0) and (not max0 and not min0) and not ext0 and not ext0:
					classifyVal = 4
				if ((max0 and (not max1 and not min1)) or ((not max0 and not min0) and max1)) and not ext0 and not ext1:
					classifyVal = 1 #5
				if ((min0 and (not max1 and not min1)) or ((not max0 and not min0) and min1)) and not ext0 and not ext1:
					classifyVal = 2 #6
				if (min0 and ext1) or (min1 and ext0):
					classifyVal = 2 #7 
				if ((not max0 and not min0) and ext1) or (ext0 and (not max1 and not min1)):
					classifyVal = 4 #8

				# Create a dictionary to append to the dataframe
				newDict = {'level': level, 'c1': self0.c1, 'c2': self0.c2, 'y0Val': averageValy0, 'y1Val': averageValy1, 'class': classifyVal}

				# Append to the dataframe
				dfNew = pd.DataFrame(newDict, index=[ii])
				df = pd.concat([df, dfNew], ignore_index = True)

				# Add to matrix
				MNew[jj, ii] = classifyVal

	# Run along the odd rows and impute as necessary
	for ii in range(1, nv1New, 2):
		for jj in range(1, nv2New, 2):
			left = MNew[jj, ii-1]
			right = MNew[jj, ii+1]
			down = MNew[jj-1, ii]
			up = MNew[jj+1, ii]
			if left == right and up == down and right == up:
				MNew[jj, ii] = MNew[jj, ii-1]

				# Create a dictionary to append to the dataframe
				newDict = {'level': level, 'c1': v1New[ii], 'c2': v2New[jj], 'y0Val': np.nan, 'y1Val': np.nan, 'class': MNew[jj, ii]}

				# Append to the dataframe
				dfNew = pd.DataFrame(newDict, index=[ii])
				df = pd.concat([df, dfNew], ignore_index = True)

			else:
				# Create new datasets
				sim0 = Simulation(nTraity=21, nTraitb2=21, c1=v1New[ii], c2=v2New[jj], ntEvo=500, yInit=0, delta=delta)
				sim1 = Simulation(nTraity=21, nTraitb2=21, c1=v1New[ii], c2=v2New[jj], ntEvo=500, yInit=1, delta=delta)
				sim0.evoSim(refDir0, True)
				sim1.evoSim(refDir1, True)

				# Extract the two files that we have just created
				createdFile0 = [f for f in os.listdir(refDir0) if os.path.isfile(os.path.join(refDir0, f))][-1][:-4]
				createdFile1 = [f for f in os.listdir(refDir1) if os.path.isfile(os.path.join(refDir1, f))][-1][:-4]
				
				# Load the data
				file0 = open('%s%s.pkl' % (refDir0,createdFile0), 'rb')
				pdict0 = pickle.load(file0)
				file0.close()
				file1 = open('%s%s.pkl' % (refDir1,createdFile1), 'rb')
				pdict1 = pickle.load(file1)
				file1.close()

				# Find the final value of y in each dataset
				self0 = pdict0["self"]
				self1 = pdict1["self"]
				yTraits0 = pdict0["yTraitListMat"][-1,:]
				yTraits1 = pdict1["yTraitListMat"][-1,:]
				store0 = pdict0['store']
				store1 = pdict1['store']
				storeFinaly0 = store0[-1, 1:(1+self0.nTraity)] + np.sum(np.reshape(store0[-1, (1+self0.nTraity+self0.nTraitb2):], (self0.nTraity, self0.nTraitb2)), axis=1)
				storeFinaly1 = store1[-1, 1:(1+self1.nTraity)] + np.sum(np.reshape(store1[-1, (1+self1.nTraity+self1.nTraitb2):], (self1.nTraity, self1.nTraitb2)), axis=1)
				if np.sum(yTraits0) > 0:
					averageValy0 = np.sum(yTraits0*self0.yTraits*storeFinaly0)/sum(storeFinaly0)
				else:
					averageValy0 = np.nan
				if np.sum(yTraits1) > 0:
					averageValy1 = np.sum(yTraits1*self1.yTraits*storeFinaly1)/sum(storeFinaly1)
				else:
					averageValy1 = np.nan

				# Series of Boolean variables to help classify
				max0 = np.abs(averageValy0-1) < 1e-3
				min0 = np.abs(averageValy0) < 1e-3
				ext0 = np.isnan(averageValy0)
				max1 = np.abs(averageValy1-1) < 1e-3
				min1 = np.abs(averageValy1) < 1e-3
				ext1 = np.isnan(averageValy1)

				# Classify
				if max0 and max1 and not ext0 and not ext1:
					classifyVal = 1
				if min0 and min1 and not ext0 and not ext1:
					classifyVal = 2
				if (max0 and min1) or (min0 and max1) and not ext0 and not ext1:
					classifyVal = 3
				if (not max0 and not min0) and (not max0 and not min0) and not ext0 and not ext0:
					classifyVal = 4
				if ((max0 and (not max1 and not min1)) or ((not max0 and not min0) and max1)) and not ext0 and not ext1:
					classifyVal = 1 #5
				if ((min0 and (not max1 and not min1)) or ((not max0 and not min0) and min1)) and not ext0 and not ext1:
					classifyVal = 2 #6
				if (min0 and ext1) or (min1 and ext0):
					classifyVal = 2 #7 
				if ((not max0 and not min0) and ext1) or (ext0 and (not max1 and not min1)):
					classifyVal = 4 #8

				# Create a dictionary to append to the dataframe
				newDict = {'level': level, 'c1': self0.c1, 'c2': self0.c2, 'y0Val': averageValy0, 'y1Val': averageValy1, 'class': classifyVal}

				# Append to the dataframe
				dfNew = pd.DataFrame(newDict, index=[ii])
				df = pd.concat([df, dfNew], ignore_index = True)

				# Add to matrix
				MNew[jj, ii] = classifyVal

	

	df.to_csv(dataFrame, index=False)

def plotClassify(dataFrame, levels=None):
	'''
	Plotting a heat map based on data from a dataframe
	If levels isn't specified, plots the highest level in the dataframe.
	If levels are specified, plots each of the levels specified.
	'''

	# Load the dataframe into a dataframe object
	df = pd.read_csv(dataFrame)

	# Obtain a list of levels present in the dataframe
	dfLevels = np.unique(df['level'].values)

	# Check that the specified levels exist and assign plotting levels
	if levels == None:
		plotLevels = np.array([np.max(dfLevels)], dtype=int)
		nL = 1
	elif set(levels).issubset(set(dfLevels)):
		plotLevels = np.array(levels, dtype=int)
		nL = len(levels)
	else:
		print('Specified levels are not all in the dataFrame, setting levels to final.')
		plotLevels = np.array([np.max(dfLevels)], dtype=int)
		nL = 1
	
	# Create a list of length nL
	storageList = [None] * nL
	v1List = [None] * nL
	v2List = [None] * nL
	dv1 = [None] * nL
	dv2 = [None] * nL
	
	# Set up the vectors and matrices for each level
	for ii in range(nL):
		
		# Find the vectors and storage
		dfL = df.loc[df['level'] <= int(plotLevels[ii])]
		nv1 = int(np.sqrt(len(dfL)))
		nv2 = int(np.sqrt(len(dfL)))
		v1List[ii] = np.linspace(0, 1, nv1)
		v2List[ii] = np.linspace(-5, 5, nv2)
		storageList[ii] = np.zeros((nv2, nv1))
		dv1[ii] = v1List[ii][1] - v1List[ii][0]
		dv2[ii] = v2List[ii][1] - v2List[ii][0]

	# Populate based on each row of the dataFrame
	for jj in range(len(df)):

		# Extract the row of the dataFrame
		dfjj = df.iloc[jj]

		# Extract the level
		dfL = dfjj['level']

		# Loop through the plotting levels
		for kk in range(nL):

			# If dfL is less than the current plotting level, add to matrix
			if dfL <= plotLevels[kk]:

				# Add to matrix
				col = int(np.round(dfjj['c1']/dv1[kk]))
				row = int(np.round((dfjj['c2']+5)/dv2[kk]))
				storageList[kk][row, col] = dfjj['class']

	# Create figures
	# List of figure handles
	figs = [None] * nL

	# Create a colourmap for the 4 categories
	cmap = ListedColormap(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'])

	# Loop through the number of levels and create a plot
	for mm in range(nL):

		# Create the figure
		fig = plt.figure()
		ax = fig.add_subplot(111)
		
		# Plot the data
		im = ax.pcolormesh(v1List[mm], v2List[mm], storageList[mm]-0.5, cmap=cmap, vmin=0, vmax=4)
		# if plotLevels[mm] >= 4:
		# 	ct=ax.contour(v1List[mm], v2List[mm], storageList[mm]-0.5, colors='k', levels=[0,1,2,3,4])
		ax.set_xlabel(r'Strength of the cost of host protection, $c_1$')
		ax.set_ylabel(r'Shape of the trade-off, $c_2$')
		ax.text(0.3, 2.0, 'Maximised', ha='center', va='center', size=14)
		ax.text(0.865, 0.0, 'Minimised', ha='center', va='center', size=14, color='w')
		ax.text(0.78, 3.0, 'CSS', ha='center', va='center', size=14)
		ax.text(0.5, -3.0, 'Repeller\n(max or min)', ha='center', va='center', size=14)
		ax.text(-0.2, -2.5, 'Decelerating', ha='center', va='center', rotation='vertical', size=14)
		ax.text(-0.2, 2.5, 'Accelerating', ha='center', va='center', rotation='vertical', size=14)
		ax.text(-0.2, 0, '-----', ha='center', va='center')

		plt.tight_layout()
		ax.set_aspect(1./ax.get_data_ratio())

	plt.show()

def singleMutSweep(ny = 51, b=0.25, alpha1=0.1, c1Vec=[0.25, 0.75], c2=2, delta=0, singleColumn=True):
	'''
	Code to loop over a range of y values to find the evolved parasite virulence, relised virulence and proportion of parasitised hosts with the defensive symbiont.

	INPUTS
	------
	ny: Number of y values equally spaced across (0,1)
	b: Background mortality
	alpha1: Defensive symbiont virulence
	c1Vec: Vector of defensive symbiont cost strengths. Can be at most 3
	c2: Defensive symbiont cost shape
	delta: Tolerance or resistance
	singleColumn: Sets whether the plot should be large enough for one column or two column manuscripts 
	'''

	# Start a figure
	if singleColumn:
		paperwidth = 3.5
		paperheight = 5.25
		matplotlib.rcParams.update({'font.size': 6})
	else:
		paperwidth = 6.4
		paperheight = 9.6
		matplotlib.rcParams.update({'font.size': 12})
	fig = plt.figure(figsize=(paperwidth, paperheight))
	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	ax1.text(-0.05, 600, '(a)')
	ax2.text(-0.05, 70, '(b)')
	ax3.text(-0.05, 0.9, '(c)')

	# Loop through the c1 vectors
	for jj in range(len(c1Vec)):
	
		# Create a vector of mutualist trait values, a vector to store the evolved virulence and a vector to store the number of deaths at steady state
		yy = np.linspace(0, 1, ny)[:-1]
		parasiteVir = np.zeros(ny-1)
		SSMat = np.zeros((4,ny-1))
		NMat = np.zeros(ny-1)
		realVir = np.zeros(ny-1)

		# Loop through the vector, initialise a simulation and find any singular strategies
		for ii in range(ny-1):
			sim = Simulation(
				nTraity = 1,
				ytraitMin = yy[ii],
				ytraitMax = yy[ii],
				b = b,
				alpha1 = alpha1,
				c1 = c1Vec[jj],
				c2 = c2,
				delta = delta
				)
			singStrats = sim.edgeStep([yy[ii], yy[ii], 0.5, 10], 51)
			singStratCoords = [singStrats[jj][1:2] for jj in range(0, len(singStrats), 2)]
			if len(singStratCoords) == 1:
				parasiteVir[ii] = singStratCoords[0]
			elif len(singStratCoords) == 0:
				# This means that the selection gradient is always positive or always negative. Find which
				max = sim.selGrad(yy[ii], 0.5)[3] > 0
				parasiteVir[ii] = 10*max + 0.5*(not max)
		
			# Calculate the steady states
			SSMat[:,ii] = sim.steadyState(yy[ii], parasiteVir[ii], sim.state0)

			# Calculate the percentage change in population size
			popBef = sum(sim.SSPar(sim.b2Init, sim.state0[[0,2]]))
			popAfter = sum(SSMat[:,ii])
			NMat[ii] = 100*(popAfter/popBef - 1)

			# Calculate the difference in realised virulence
			virBef = sim.alpha2fun(sim.b2Init)*sim.SSPar(sim.b2Init, sim.state0[[0,2]])[1]/popBef
			virAfter = (sim.alpha1*SSMat[1,ii] + sim.alpha2fun(parasiteVir[ii])*SSMat[2,ii] + sim.alpha12fun(yy[ii], parasiteVir[ii])*SSMat[3,ii])/popAfter
			realVir[ii] = 100*(1 - virAfter/virBef)

		# Plot colour and linestyle
		col = 'k'*(jj == 0) + 'r'*(jj == 1) + 'b'*(jj == 2)
		sty = '--'*(jj == 0) + '-'*(jj == 1) + ':'*(jj == 2)

		# Plot the relative increase in virulence
		ax1.plot(yy, 100*(sim.alpha2fun(parasiteVir)/sim.alpha2fun(sim.b2Init)-1), lw=2, c=col, ls=sty, label=r'$c_1=$' + str(c1Vec[jj]))
		ax1.set_xticks([0, 0.5, 1]) 
		ax1.set_xticklabels([])
		ax1.set_ylabel('% increase in\nevolved virulence')

		# Set the left bound of each axis to 0
		xx = ax1.get_xlim()
		xx = (0, xx[1])
		ax1.set_xlim(xx)
		yys = ax1.get_ylim()
		yys = (0, yys[1])
		ax1.set_ylim(yys)
		
		# Realised virulence
		ax2.plot(yy, -realVir, lw=2, c=col, ls=sty, label=r'$c_1=$' + str(c1Vec[jj]))
		ax2.set_xticks([0, 0.5, 1])
		ax2.set_xticklabels([])
		ax2.set_ylabel('% increase in average\nrealised virulence')

		# Set the left bound of each axis to 0
		xx = ax2.get_xlim()
		xx = (0, xx[1])
		ax2.set_xlim(xx)
		yys = ax2.get_ylim()
		yys = (0, yys[1])
		ax2.set_ylim(yys)

		# Plot the ratio of coinfections to single infections
		ax3.plot(yy, SSMat[3,]/(SSMat[2,]+SSMat[3,]), lw=2, c=col, ls=sty, label=r'$c_1=$' + str(c1Vec[jj]))
		ax3.set_xticks([0, 0.5, 1])
		ax3.set_xlabel(r'Strength of host protection, $y$')
		ax3.set_ylabel('Proportion of parasitised\nhosts that posesss\ndefensive symbionts')

		# Set the left bound of each axis to 0
		xx = ax3.get_xlim()
		xx = (0, xx[1])
		ax3.set_xlim(xx)
		yys = ax3.get_ylim()
		yys = (0, yys[1])
		ax3.set_ylim(yys)

	ax1.legend()
	plt.tight_layout()
	plt.savefig('./Fig2.pdf')
	plt.show() 

def newEvoSimGif(dataDir, saveDir, figNum, c1=None, c2=None, yInit=[0.1,0.5,0.9]):
	'''
	Generates several figures based on the data stored in dataDir.
	If dataDir already exists, the information is plotted, otherwise, data is first created and then plotted.
	figNum is the order with which to plot in the 2x2 array. Should be an integer between 0 and 3
	Outputs an evo sim for each, plus a figure which has the two heat maps eith trajectories
	'''

	# Check if the data directory exists. If it doesn't create it
	isDir = os.path.isdir(dataDir)
	if not isDir:
		os.mkdir(dataDir)

	# Check if the save directory exists. If it doesn't create it
	isSaveDir = os.path.isdir(saveDir)
	if not isSaveDir:
		os.mkdir(saveDir)

	# Default c1
	if c1 == None:
		c1 = 0.25
	
	# Default c2
	if c2 == None:
		c2 = 2

	# If the directory didn't exist before, we run the data
	if not isDir:

		# Loop through the init values
		for ii in range(len(yInit)):
			print(ii)

			sim = Simulation(c1=c1, c2=c2, yInit=yInit[ii])
			sim.evoSim(dataDir, saveData=True)

	# Find the number of plots in the data dir
	filesInDir = [f for f in os.listdir(dataDir) if os.path.isfile(os.path.join(dataDir, f))]
	nFiles = len(filesInDir)

	# Open the first simulation file
	file = open(dataDir + filesInDir[0], 'rb')
	pdict = pickle.load(file)
	file.close()
	self = pdict['self']

	# Create storage arrays
	deathRateMat = np.zeros((self.nTraitb2, self.nTraity))
	NMat = np.zeros((self.nTraitb2, self.nTraity))
	yAveLow = np.zeros((int(np.round(self.ntEvo/10+1)), nFiles)) # Every tenth time point
	yAveUp = np.zeros((int(np.round(self.ntEvo/10+1)), nFiles)) # Every tenth time point
	b2AveLow = np.zeros((int(np.round(self.ntEvo/10+1)), nFiles)) # Every tenth time point
	b2AveUp = np.zeros((int(np.round(self.ntEvo/10+1)), nFiles)) # Every tenth time point

	# Generate the data for the heat maps
	for ii in range(self.nTraity):
		for jj in range(self.nTraitb2):
			SS = self.steadyState(self.yTraits[ii], self.b2Traits[jj], self.state0)
			N = np.sum(SS)
			SSPar = self.SSPar(self.b2Traits[jj], self.state0[[0,2]])
			NPar = np.sum(SSPar)
			deathRateMat[jj,ii] = 1 - (self.alpha1fun(self.yTraits[ii])*SS[1]/N + self.alpha2fun(self.b2Traits[jj])*SS[2]/N + self.alpha12fun(self.yTraits[ii], self.b2Traits[jj])*SS[3]/N)/(self.alpha2fun(self.b2Traits[jj])*SSPar[1]/NPar)
			NMat[jj,ii] = np.sum(self.steadyState(self.yTraits[ii], self.b2Traits[jj], self.state0))

	# Loop through each of the datasets
	for n in range(nFiles):

		# Open file and extract
		file = open(dataDir + filesInDir[n], 'rb')
		pdict = pickle.load(file)
		file.close()
		self = pdict['self']
		yInds = pdict['yTraitListMat']
		b2Inds = pdict['b2TraitListMat']
		store = pdict['store']

		# Initialise
		yAveLow[0,n] = self.yInit
		yAveUp[0,n] = self.yInit
		b2AveLow[0,n] = self.b2Init
		b2AveUp[0,n] = self.b2Init

		# Find the yStore and b2Store values
		IIy = store[:,1:(1+self.nTraity)]
		IIb2 = store[:,(1+self.nTraity):(1+self.nTraity+self.nTraitb2)]
		for t in range(self.ntEvo+1):
			I12 = np.reshape(store[t, (1+self.nTraity+self.nTraitb2):], (self.nTraity, self.nTraitb2))
			IIy[t,] += np.sum(I12, axis=1)
			IIb2[t,] += np.sum(I12, axis=0)

		# Loop through the evo timesteps
		for jj in range(1, int(np.round(self.ntEvo/10+1))):

			# Index from main dataset
			ii = jj*10

			# Find the average values
			# Start with finding the non-zero indices
			yTraits = yInds[ii, :]
			yTraits = yTraits*np.arange(1, self.nTraity+1)
			yTraits = (yTraits[yTraits > 0] - 1).astype('int')
			b2Traits = b2Inds[ii, :]
			b2Traits = b2Traits*np.arange(1, self.nTraitb2+1)
			b2Traits = (b2Traits[b2Traits > 0] - 1).astype('int')

			# Check if there are two branches in the parasite
			branch = np.diff(b2Traits)
			branch = np.sum(branch > 1) > 0

			# If there is a branch we need to save the different averages
			if branch:
				
				# Find the splits
				b2Low = [b2Traits[0]]
				b2Up = []
				lower = b2Traits[1] - b2Traits[0] == 1
				for kk in range(1, len(b2Traits)):
					if lower:
						b2Low.append(b2Traits[kk])
						lower = b2Traits[kk+1] - b2Traits[kk] == 1
					else:
						b2Up.append(b2Traits[kk])

				yAveLow[jj,n] = np.sum(self.yTraits[yTraits]*IIy[ii, yTraits]/sum(IIy[ii, yTraits]))
				yAveUp[jj,n] = np.sum(self.yTraits[yTraits]*IIy[ii, yTraits]/sum(IIy[ii, yTraits]))
				b2AveLow[jj,n] = np.sum(self.b2Traits[b2Low]*IIb2[ii, b2Low]/sum(IIb2[ii, b2Low]))
				b2AveUp[jj,n] = np.sum(self.b2Traits[b2Up]*IIb2[ii, b2Up]/sum(IIb2[ii, b2Up]))

			else:
				yAveLow[jj,n] = np.sum(self.yTraits[yTraits]*IIy[ii, yTraits]/sum(IIy[ii, yTraits]))
				yAveUp[jj,n] = np.sum(self.yTraits[yTraits]*IIy[ii, yTraits]/sum(IIy[ii, yTraits]))
				b2AveLow[jj,n] = np.sum(self.b2Traits[b2Traits]*IIb2[ii, b2Traits]/sum(IIb2[ii, b2Traits]))
				b2AveUp[jj,n] = np.sum(self.b2Traits[b2Traits]*IIb2[ii, b2Traits]/sum(IIb2[ii, b2Traits]))


	# Colourmaps
	cmap = ListedColormap(['#ef8a62','#fddbc7','#d1e5f0','#67a9cf'])
	cmap.set_under('#b2182b')
	cmap.set_over('#2166ac')
	cmapalt = ListedColormap(['#af8dc3','#e7d4e8','#d9f0d3','#7fbf7b'])
	cmapalt.set_under('#762a83')
	cmapalt.set_over('#1b7837')
	matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

	# Create the first figure. Two heatmaps with trajectories
	plotWidth = 11.69
	plotHeight = 8.27
	fig1 = plt.figure(figsize=(plotWidth, plotHeight))
	ax11 = fig1.add_subplot(121)
	ax12 = fig1.add_subplot(122)

	# First axis, which is the population size comparison
	NParCSS = np.sum(self.SSPar(self.b2Init, self.state0[[0,2]]))
	ax11.contourf(self.yTraits, self.b2Traits, 100*(NMat/NParCSS-1), levels=[-50,-25,0,25,50], extend='both', cmap=cmapalt, alpha=0.3)
	contour11 = ax11.contour(self.yTraits, self.b2Traits, 100*(NMat/NParCSS-1), levels=[-50,-25,0,25,50], colors='dimgray')
	ax11.clabel(contour11, inline=True, fontsize=18)
	ax11.plot([0,1], [self.b2Init, self.b2Init], 'k--')
	for n in range(nFiles):
		ax11.plot(yAveLow[:,n], b2AveLow[:,n], 'k', lw=2)
		ax11.plot(yAveUp[:,n], b2AveUp[:,n], 'k', lw=2)
		ax11.plot(yAveLow[0,n], b2AveLow[0,n], 'g.', ms=18)
		ax11.plot(yAveLow[-1,n], b2AveLow[-1,n], 'r.', ms=18)
		ax11.plot(yAveUp[0,n], b2AveUp[0,n], 'g.', ms=18)
		ax11.plot(yAveUp[-1,n], b2AveUp[-1,n], 'r.', ms=18)
	ax11.set_xlabel(r'$y$', fontsize=18)
	ax11.set_ylabel(r'$\beta_P$', fontsize=18)
	ax11.set_title('Population size', fontsize=18)
	ax11.tick_params(axis='x', labelsize=18)
	ax11.tick_params(axis='y', labelsize=18)

	# Second axis, which is the death rate
	ax12.contourf(self.yTraits, self.b2Traits, gaussian_filter(100*deathRateMat, 1), levels=[-20,-10,0,10,20], extend='both', cmap=cmap, alpha=0.3)
	contour12 = ax12.contour(self.yTraits, self.b2Traits, gaussian_filter(100*deathRateMat, 1), levels=[-20,-10,0,10,20], colors='dimgray')
	ax12.clabel(contour12, inline=True, fontsize=18)
	ax12.plot([0,1], [self.b2Init, self.b2Init], 'k--')
	for n in range(nFiles):
		ax12.plot(yAveLow[:,n], b2AveLow[:,n], 'k', lw=2)
		ax12.plot(yAveUp[:,n], b2AveUp[:,n], 'k', lw=2)
		ax12.plot(yAveLow[0,n], b2AveLow[0,n], 'g.', ms=18)
		ax12.plot(yAveLow[-1,n], b2AveLow[-1,n], 'r.', ms=18)
		ax12.plot(yAveUp[0,n], b2AveUp[0,n], 'g.', ms=18)
		ax12.plot(yAveUp[-1,n], b2AveUp[-1,n], 'r.', ms=18)
	ax12.set_xlabel(r'$y$', fontsize=18)
	ax12.set_yticklabels('')
	ax12.set_title('Death Rate', fontsize=18)
	ax12.tick_params(axis='x', labelsize=18)

	plt.tight_layout()
	pdict = {'yT': self.yTraits, 'bT':self.b2Traits, 'deathRateMat': deathRateMat, 'NMat': NMat/NParCSS-1, 'cmap':cmap, 'cmapalt': cmapalt, 'yL': yAveLow, 'yU': yAveUp, 'bL': b2AveLow, 'bU': b2AveUp}
	file = open(saveDir + str(figNum) + '_Data_' + str(self.c1) + '_' + str(self.c2) + '.pkl', 'wb')
	pickle.dump(pdict, file)
	file.close()
	fig1.clf()
	plt.close()

	