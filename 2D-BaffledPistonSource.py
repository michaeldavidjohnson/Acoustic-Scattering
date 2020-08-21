from scipy.integrate import quad,romberg
from scipy.misc import derivative
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
import time

'''Undirected2D is a class used to calculate the Kirchoff Approximation on any given functional surface.
    The source that the user defines the location is assumed to be an undirected point source. This means that
    the acoustic pressure released from the source is released uniformly acting like a circle. The kirchoff approximation
    only works on specific surfaceFunctions. There is a method in this class which checks if your surface is appropriate for this
    algorithm.'''
class BaffledPiston2DNew:
    def __init__(self, sourceLocations, receiverLocations, frequency, surfaceFunction, surfaceLength,sourceAngle = -np.pi/4,a=0.04/2):
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
        self.ayy = a
        self.sourceAngle = sourceAngle
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
    def AngleChecker(self,xMin=-20,xMax=20,results = False):
        def isNaN(num):
            return num != num
        self.Singularities = []
        AngleArray = []
        for x in np.linspace(xMin,xMax,100_000):
            elev = self.surfaceFunction(x)
            AC = [1+self.sourceLocations[0]-self.sourceLocations[0],0]
            AB = [x-self.sourceLocations[0],elev-self.sourceLocations[1]]
            angle =  - np.arccos((AB[0]*AC[0]+AB[1]*AC[1])/(np.sqrt(AC[0]**2 +AC[1]**2)*np.sqrt(AB[0]**2+AB[1]**2)))
            angle -= self.sourceAngle
            if np.abs(np.sin(angle)) < 0.01:
                self.Singularities.append(x)
        if results == False:
            return 0 
        else:
            return(self.Singularities)

    def CalculateAngle(self,elev, x):
        AC = [1+self.sourceLocations[0]-self.sourceLocations[0],0]
        AB = [x-self.sourceLocations[0],elev-self.sourceLocations[1]]
        angle =  - np.arccos((AB[0]*AC[0]+AB[1]*AC[1])/(np.sqrt(AC[0]**2 +AC[1]**2)*np.sqrt(AB[0]**2+AB[1]**2)))
        angle -= self.sourceAngle
        return angle

    def CalculateDirectivity(self,angle,k,ayy):
        if np.abs(angle) < 0.001:
            Directivity = 1
        else:
            Directivity = (2*sp.special.jn(1,k*ayy*np.sin(angle)))/(k*ayy*np.sin(angle))
        return Directivity

    def CalculateRandQ(self,sourceX,sourceZ,receiverX,receiverZ,x,k):
        a = np.sqrt((sourceX - x)**2 + (sourceZ)**2)
        b = np.sqrt((receiverX - x)**2 + (receiverZ)**2)
        qz = k*(sourceZ/a + receiverZ/b)
        return a,b,qz

    def EvaluateIntegrand(self, x, sourceX, sourceZ, receiverX, receiverZ, surfaceElevation, k,ayy, isReal = True,isRomb=False):
        elev = surfaceElevation(x)
        angle = self.CalculateAngle(elev, x)
        Directivity = self.CalculateDirectivity(angle,k,ayy)
        a,b,qz = self.CalculateRandQ(sourceX,sourceZ,receiverX,receiverZ,x,k)
        if isRomb == True:
            return (Directivity / np.sqrt(a * b)) * np.exp(0 + 1j*(k*(a + b)-qz*elev ) )*(qz) #Romberg method in scipy can handle complex number as input.
        elif isReal == True:
            return sp.real((Directivity / np.sqrt(a * b)) * np.exp(0 + 1j*( k*(a + b)-qz*elev ) )*(qz)) #As quad cannot handle complex numbers it needs to be split up into real and imaginary component.

        else:
            return sp.imag((Directivity / np.sqrt(a * b)) * np.exp(0 + 1j*( k*(a + b)-qz*elev ) )*(qz))  #Integrand from the KA


    ''' This method calculates KA, for both one or more receivers.'''
    def UseQuad(self,receiverLocations):
        Min = -np.infty
        Max = np.infty
        Real = (quad(self.EvaluateIntegrand, Min, Max, args = (self.sourceLocations[0], self.sourceLocations[1], receiverLocations[0], receiverLocations[1],
                                                         self.surfaceFunction, self.k,self.ayy,True,False),epsabs = self.error,epsrel = self.error, limit = self.limit))
        Imaginary = (quad(self.EvaluateIntegrand, Min, Max, args = (self.sourceLocations[0], self.sourceLocations[1], receiverLocations[0], receiverLocations[1],
                                                              self.surfaceFunction, self.k,self.ayy,False,False),epsabs = self.error, epsrel = self.error, limit = self.limit))
        temp = -1j / (2 * np.pi * self.k ) * (Real[0] + 1j * Imaginary[0]) #The actual answer for KA
        return temp

    '''This method will calculate the Kirchoff Approximation algorithm for one or more recievers by using the Romberg integration
      as per SciPy's documentation. If your scattered field is taking a rather long time to get the required accuracy for your 
      2D function. Then maybe this method will converge faster. It's highly possible that Romberg may be slower but because Quad
      is called twice at every step. Romberg may be slightly faster. For some of my testing I could significantly reduce my time 
      using this method. Quad will always be kept default due to nquad being the main method for 3D integration.'''
    def UseRomb(self,receiverLocations):
        Min = -20
        Max = 20
        divmax = 40
        romb = (romberg(self.EvaluateIntegrand,Min,Max,args = (self.sourceLocations[0], self.sourceLocations[1], receiverLocations[0], receiverLocations[1],
                                                                          self.surfaceFunction, self.k,self.ayy,False,True),tol=self.error,rtol = self.error,divmax=divmax))
        temp = -1j / (2 * np.pi * self.k ) * (romb) #The actual answer for KA
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

    def Scatter(self, graphs = False,dbTest=False, results = True,useQuad = True,useRomb = False,error = 1e-6,limit=2500,normalised = True):
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
                if normalised == True:
                    axes[0].plot(np.array(self.receiverLocations).T[1], self.RealResults / np.max(self.AbsoluteResults)) #Normalised by the absolute results
                    axes[1].plot(np.array(self.receiverLocations).T[1], self.ImaginaryResults / np.max(self.AbsoluteResults))
                    axes[2].plot(np.array(self.receiverLocations).T[1], self.AbsoluteResults / np.max(self.AbsoluteResults))
                else:
                    axes[0].plot(np.array(self.receiverLocations).T[1], self.RealResults) #Normalised by the absolute results
                    axes[1].plot(np.array(self.receiverLocations).T[1], self.ImaginaryResults)
                    axes[2].plot(np.array(self.receiverLocations).T[1], self.AbsoluteResults)
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
                if normalised == True:
                    axes[0].plot(np.array(self.receiverLocations).T[0], self.RealResults / np.max(self.AbsoluteResults),color = self.plotColour)
                    axes[1].plot(np.array(self.receiverLocations).T[0], self.ImaginaryResults / np.max(self.AbsoluteResults),color = self.plotColour)
                    axes[2].plot(np.array(self.receiverLocations).T[0], self.AbsoluteResults / np.max(self.AbsoluteResults),color = self.plotColour) 
                else:
                    axes[0].plot(np.array(self.receiverLocations).T[0], self.RealResults,color = self.plotColour)
                    axes[1].plot(np.array(self.receiverLocations).T[0], self.ImaginaryResults,color = self.plotColour)
                    axes[2].plot(np.array(self.receiverLocations).T[0], self.AbsoluteResults,color = self.plotColour) 
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
                if normalised == True:
                    axes.plot(np.array(self.receiverLocations).T[0], np.array(self.receiverLocations.T)[1], self.RealResults / np.max(self.AbsoluteResults),color = self.plotColour)
                else:
                    axes.plot(np.array(self.receiverLocations).T[0], np.array(self.receiverLocations.T)[1], self.RealResults,color = self.plotColour)
                axes.set_xlabel('x')
                axes.set_ylabel('y')
                axes.set_zlabel('Real Pressure')
                axes = fig.add_subplot(1,3,2,projection='3d')
                if normalised == True:
                    axes.plot(np.array(self.receiverLocations.T)[0], np.array(self.receiverLocations.T)[1], self.ImaginaryResults / np.max(self.AbsoluteResults),color = self.plotColour)
                else:
                    axes.plot(np.array(self.receiverLocations.T)[0], np.array(self.receiverLocations.T)[1], self.ImaginaryResults,color = self.plotColour)
                axes.set_xlabel('x')
                axes.set_ylabel('y')
                axes.set_zlabel('Imaginary Pressure')
                axes = fig.add_subplot(1,3,3,projection='3d')
                if normalised == True:
                    axes.plot(np.array(self.receiverLocations).T[0], np.array(self.receiverLocations.T)[1], self.AbsoluteResults / np.max(self.AbsoluteResults),color = self.plotColour)
                else:
                    axes.plot(np.array(self.receiverLocations).T[0], np.array(self.receiverLocations.T)[1], self.AbsoluteResults,color = self.plotColour)
                axes.set_xlabel('x')
                axes.set_ylabel('y')
                axes.set_zlabel('Absolute Pressure')
        if dbTest == True:
            self.AbsoluteResults = self.AbsoluteResults / np.max(self.AbsoluteResults)
            return self.AbsoluteResults

        return self.Results
