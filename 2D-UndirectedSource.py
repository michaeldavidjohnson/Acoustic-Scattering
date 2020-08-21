from scipy.integrate import quad,romberg
from scipy.misc import derivative
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D

'''Undirected2D is a class used to calculate the Kirchoff Approximation on any given functional surface.
    The source that the user defines the location is assumed to be an undirected point source. This means that
    the acoustic pressure released from the source is released uniformly acting like a circle. The kirchoff approximation
    only works on specific surfaceFunctions. There is a method in this class which checks if your surface is appropriate for this
    algorithm.'''
class Undirected2D:
    def __init__(self, sourceLocations, receiverLocations, frequency, surfaceFunction, surfaceLength):
        if np.array(sourceLocations).shape == (2,): #Catching the correct shape.
            self.sourceLocations = np.array(sourceLocations) #Transfering it to numpy arrays for easier unpacking.
        else:    
            raise Exception("The input is incorrect, sourceLocations is expecting an input of the form (a,b)")

        if np.array(receiverLocations).shape == (2,) or np.array(receiverLocations).shape[1] == 2: #Catching the correct shape.
            self.receiverLocations = np.array(receiverLocations) #Numpy arrays are better to unpack
        else:
            raise Exception("The input is incorrect, receiverLocations is expecting an input of the form ((a,b),...(an,bn)) where n is the amount of receivers.")

        self.k = (2 * np.pi * frequency) / 343 #Converting the frequency of the source to the appropriate wavenumber
        self.surfaceFunction = surfaceFunction #Should probably have some checking that this is actually a function.

        if np.array(surfaceLength).shape == (2,): #Catching the correct shape.
            self.surfaceLength = surfaceLength #This will eventually be used in the integration limits.
        else:
            raise Exception("The input is incorrect, surfaceLength is expecting an input of the form (a,b)")

        self.Results = []
        self.RealResults = []
        self.ImaginaryResults = []
        self.AbsoluteResults = []
        self.plotColour = 'blue'
    def SetPlotColour(self,col):
        self.plotColour = col

    '''This method will check if the surface you are using satisfies the necessary conditions required for Kirchoff Approximation to hold.'''
    def SurfaceChecker(self, relaxed = True):
        conditionData = [] #Storing ready to check the condition
        xMin, xMax = self.surfaceLength #Differentiating along the length of the surface.
        xStep = 0.001 #Could make this user defined.

        for i in np.arange(xMin, xMax, xStep):
            dx = derivative(self.surfaceFunction, i, dx=1e-6, n=1) #Single derivative
            dxx = derivative(self.surfaceFunction, i, dx=1e-6, n=2) #Double derivative
            a = np.abs((1 + (dx)**2)**1.5 / (dxx)) #Radius of curvature for 2D
            conditionToTest = 1 / ((self.k * a)**0.33333333333) #Kirchoff approximation condition

            if relaxed == True:
                if conditionToTest < 1: #Relaxed condition given by a recent article.
                    conditionData.append(conditionToTest) 
                else:
                    raise Exception("The surface does not satisfy the conditions for Kirchoff Approximation")
            else:
                if conditionToTest < 0.1: #The original condition
                    conditionData.append(conditionToTest)
                else:
                    raise Exception("The surface does not satisfy the conditions for Kirchoff Approximation")

        if len(conditionData) == len(np.arange(xMin, xMax, xStep)): #Strange checker
            print("Good choice of function")
            return True
        else:
            return False

    ''' This method evaluates the real or imaginary component of the integrand.'''
    def EvaluateIntegrand(self, x, sourceX, sourceZ, receiverX, receiverZ, surfaceElevation, k, isReal = True,isRomb=False):
        elev = surfaceElevation(x)
        delev = derivative(surfaceElevation, x, n=1, dx=1e-6)
        a = np.sqrt((sourceX - x)**2 + (sourceZ - elev)**2)
        b = np.sqrt((receiverX - x)**2 + (receiverZ - elev)**2)
        qx = -k * ((-receiverX + x) / (np.sqrt((receiverX - x)**2 + (receiverZ - elev)**2)) + (-sourceX + x) / (np.sqrt((sourceX - x)**2 + (sourceZ - elev)**2)))
        qz = -k * ((-receiverZ + elev) / (np.sqrt((receiverX - x)**2 + (receiverZ - elev)**2)) + (-sourceZ + elev) / (np.sqrt((sourceX - x)**2 + (sourceZ - elev)**2)))
        if isRomb == True:
            return (1 / (a * b)) * np.exp(0 + 1j * k * (a + b)) * (qz-delev * qx) #Romberg method in scipy can handle complex number as input.
        
        elif isReal == True:
            return sp.real((1 / (a * b)) * np.exp(0 + 1j * k * (a + b)) * (qz-delev * qx)) #As quad cannot handle complex numbers it needs to be split up into real and imaginary component.
        else:    
            return sp.imag((1 / (a * b)) * np.exp(0 + 1j * k * (a + b)) * (qz - delev * qx))  #Integrand from the KA
    ''' This method calculates KA, for both one or more receivers.'''
    def UseQuad(self,receiverLocations):
        Real = quad(self.EvaluateIntegrand, -20, 20, args = (self.sourceLocations[0], self.sourceLocations[1], receiverLocations[0], receiverLocations[1],
                                                         self.surfaceFunction, self.k,True),epsabs = self.error,epsrel = self.error, limit = self.limit) #Numerical Integration
        Imaginary = quad(self.EvaluateIntegrand, -20, 20, args = (self.sourceLocations[0], self.sourceLocations[1], receiverLocations[0], receiverLocations[1],
                                                              self.surfaceFunction, self.k,False),epsabs = self.error, epsrel = self.error, limit = self.limit )
        temp = 1 / (4j * np.pi) * (Real[0] + 1j * Imaginary[0]) #The actual answer for KA
        return temp
    
    '''This method will calculate the Kirchoff Approximation algorithm for one or more recievers by using the Romberg integration
      as per SciPy's documentation. If your scattered field is taking a rather long time to get the required accuracy for your 
      2D function. Then maybe this method will converge faster. It's highly possible that Romberg may be slower but because Quad
      is called twice at every step. Romberg may be slightly faster. For some of my testing I could significantly reduce my time 
      using this method. Quad will always be kept default due to nquad being the main method for 3D integration.'''
    def UseRomb(self,receiverLocations):
        romb = romberg(self.EvaluateIntegrand,-20,20,args = (self.sourceLocations[0], self.sourceLocations[1], receiverLocations[0], receiverLocations[1],
                                                                          self.surfaceFunction, self.k,False,True),tol=self.error,rtol = self.error,divmax=20)
        temp = 1 / (4j * np.pi) * (romb) #The actual answer for KA
        return temp
    
    '''This method will calculate the results from the Kirchoff Approximation and then store it in an array.'''
    def CalculateKA(self,i = 0):
        receiverLocations = self.receiverLocations[i]
        if self.useQuad == True:
            temp = self.UseQuad(receiverLocations)
        if self.useRomb == True:
            temp = self.UseRomb(receiverLocations)
        self.Results.append(temp)
        self.RealResults.append(sp.real(temp))
        self.ImaginaryResults.append(sp.imag(temp))
        self.AbsoluteResults.append(np.abs(temp))
        
        if self.results == True:
            print("The pressure value at ", receiverLocations, "is: ", temp) #Print the values, looks lovely but will spam your command line.

    '''This is where the main logic of the scattering method is used. graphs is used to check if the user wants graphs
        or not. Some graph specific member functions will be added such as layout options/colour/titles. Results will print out the pressure values at a receiver.'''
    def Scatter(self, graphs = False, results = True,useQuad = True,useRomb = False,error = 1e-2,limit=2500):
        self.results = results
        self.useQuad = useQuad
        self.useRomb = useRomb
        self.error = error
        self.limit = limit
        for receiverIndex in range(self.receiverLocations.shape[0]): #Logic for multiple receivers
            self.CalculateKA(receiverIndex)
        if graphs == True:
            if np.all(self.receiverLocations == self.receiverLocations[0, 0],axis = 0)[0] == True: #This checks if all of either the x or the y axis is the same for the receivers. For a line array this means we can drop a dimension in the graph.
                fig = plt.figure()
                axes = fig.subplots(3)
                axes[0].plot(np.array(self.receiverLocations).T[1], self.RealResults / np.max(self.AbsoluteResults)) #Normalised by the absolute results
                axes[1].plot(np.array(self.receiverLocations).T[1], self.ImaginaryResults / np.max(self.AbsoluteResults))
                axes[2].plot(np.array(self.receiverLocations).T[1], self.AbsoluteResults / np.max(self.AbsoluteResults))
                plt.tight_layout()
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('Real Pressure')
                axes[1].set_xlabel('x')
                axes[1].set_ylabel('Imaginary Pressure')
                axes[2].set_xlabel('x')
                axes[2].set_ylabel('Absolute Pressure')
                fig.show()
            if np.all(self.receiverLocations == self.receiverLocations[0, 1],axis = 0)[1] == True: #3D plots for when we cannot drop a dimension
                fig = plt.figure()
                axes = fig.subplots(3)
                print((self.RealResults / np.max(self.AbsoluteResults)).shape)
                axes[0].plot(np.array(self.receiverLocations).T[0], self.RealResults / np.max(self.AbsoluteResults),color = self.plotColour)
                axes[1].plot(np.array(self.receiverLocations).T[0], self.ImaginaryResults / np.max(self.AbsoluteResults),color = self.plotColour)
                axes[2].plot(np.array(self.receiverLocations).T[0], self.AbsoluteResults / np.max(self.AbsoluteResults),color = self.plotColour) 
                plt.tight_layout()
                axes[0].set_xlabel('x')
                axes[0].set_ylabel('Real Pressure')
                axes[1].set_xlabel('x')
                axes[1].set_ylabel('Imaginary Pressure')
                axes[2].set_xlabel('x')
                axes[2].set_ylabel('Absolute Pressure')
                fig.show()
            else:
                fig = plt.figure()
                axes = fig.add_subplot(1,3,1,projection='3d')
                axes.plot(np.array(self.receiverLocations).T[0], np.array(self.receiverLocations.T)[1], self.RealResults / np.max(self.AbsoluteResults),color = self.plotColour)
                axes.set_xlabel('x')
                axes.set_ylabel('y')
                axes.set_zlabel('Real Pressure')
                axes = fig.add_subplot(1,3,2,projection='3d')
                axes.plot(np.array(self.receiverLocations.T)[0], np.array(self.receiverLocations.T)[1], self.ImaginaryResults / np.max(self.AbsoluteResults),color = self.plotColour)
                axes.set_xlabel('x')
                axes.set_ylabel('y')
                axes.set_zlabel('Imaginary Pressure')
                axes = fig.add_subplot(1,3,3,projection='3d')
                axes.plot(np.array(self.receiverLocations).T[0], np.array(self.receiverLocations.T)[1], self.AbsoluteResults / np.max(self.AbsoluteResults),color = self.plotColour)
                axes.set_xlabel('x')
                axes.set_ylabel('y')
                axes.set_zlabel('Absolute Pressure')
                
                
        return self.Results
