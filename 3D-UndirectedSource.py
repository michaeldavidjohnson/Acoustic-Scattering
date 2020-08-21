from scipy.integrate import quad
from scipy.misc import derivative
import numpy as np
import scipy as sp
import types
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D

class Undirected3DSlow:
    '''Undirected3D is a class used to calculate the Kirchoff Approximation on any given functional surface.
    The source that the user defines the location is assumed to be an undirected point source. This means that
    the acoustic pressure released from the source is released uniformly acting like a circle. The kirchoff approximation
    only works on specific surfaceFunctions. There is a method in this class which checks if your surface is appropriate for this
    algorithm. The reason this class has Slow at the end of it is because at the present moment in time. When you end up using the numerical
    integration from this method the runtime is extremely long. There are methods to speed this integration up significantly, 
    which will be worked on after the directivity classes occur.'''
    def __init__(self,sourceLocation,recieverLocations,frequency,surfaceFunction):
        if np.array(sourceLocation).shape == (3,): #Catching the correct shape.
            self.sourceLocation = np.array(sourceLocation) #Transfering it to numpy arrays for easier unpacking.
        else:    
            raise Exception("The input is incorrect, sourceLocation is expecting an input of the form (a,b)")
        if np.array(recieverLocations).shape == (3,) or np.array(recieverLocations).shape[1] == 2: #Catching the correct shape.
            self.recieverLocations = np.array(recieverLocations) #Numpy arrays are better to unpack
        else:
            raise Exception("The input is incorrect, recieverLocations is expecting an input of the form ((a,b),...(an,bn)) where n is the amount of recievers.")
        self.k = (2*np.pi*frequency)/343 #Converting the frequency of the source to the appropriate wavenumber
        self.surfaceElevation = surfaceFunction #Should probably have some checking that this is actually a function.
        self.Results = []
        self.RealResults = []
        self.ImaginaryResults = []
        self.AbsoluteResults = []
    
    def surfaceChecker(self,relaxed = True):
        return 0 
    
    def partial_derivative(self,func, var=0, point=[]):
        args = point[:]
        def wraps(x):
            args[var] = x
            return func(*args)
        return derivative(wraps, point[var], dx = 1e-6)
    
    def R13D(self,source_x,source_y,source_z,x,y,z):
        return np.sqrt((source_x-x)**2+(source_y-y)**2+(source_z-z)**2)
    def R23D(self,reciever_x,reciever_y,reciever_z,x,y,z):
        return np.sqrt((reciever_x-x)**2+(reciever_y-y)**2+(reciever_z-z)**2)
    
    def q3D(self,source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,x,y,surfaceElevation):
        qx = ((x-reciever_x)+(x-source_x))/(np.sqrt((reciever_x-x)**2+(reciever_y-y)**2+(reciever_z-surfaceElevation(y,x))**2))
        qy = ((y-reciever_y)+(y-source_y))/(np.sqrt((reciever_x-x)**2+(reciever_y-y)**2+(reciever_z-surfaceElevation(y,x))**2))
        qz = ((surfaceElevation(y,x)-reciever_z)+(surfaceElevation(y,x)-source_z))/(np.sqrt((reciever_x-x)**2+(reciever_y-y)**2+(reciever_z-surfaceElevation(y,x))**2))
        return [qx,qy,qz]
    
    
    def Integrand3DReal(self,x,y,source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,surfaceElevation,k):
        a = self.R13D(source_x,source_y,source_z,x,y,surfaceElevation(x,y))
        b = self.R23D(reciever_x,reciever_y,reciever_z,x,y,surfaceElevation(x,y))
        qx = -k*self.q3D(source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,x,y,surfaceElevation)[0]
        qy = -k*self.q3D(source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,x,y,surfaceElevation)[1]
        qz = -k*self.q3D(source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,x,y,surfaceElevation)[2]
        elev = surfaceElevation(x,y)
        delev = [self.partial_derivative(surfaceElevation,1,[x,y]),self.partial_derivative(surfaceElevation,1,[x,y])]
        return sp.real((1/(a*b)*np.exp(0+1j*(a+b))*(qz-(delev[0]*qx+delev[1]*qy))))
    
    def Integrand3DImag(self,x,y,source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,surfaceElevation,k):
        a = self.R13D(source_x,source_y,source_z,x,y,surfaceElevation(x,y))
        b = self.R23D(reciever_x,reciever_y,reciever_z,x,y,surfaceElevation(x,y))
        qx = -k*self.q3D(source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,x,y,surfaceElevation)[0]
        qy = -k*self.q3D(source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,x,y,surfaceElevation)[1]
        qz = -k*self.q3D(source_x,source_y,source_z,reciever_x,reciever_y,reciever_z,x,y,surfaceElevation)[2]
        elev = surfaceElevation(x,y)
        delev = [self.partial_derivative(surfaceElevation,1,[y,x]),self.partial_derivative(surfaceElevation,0,[y,x])]
        return sp.imag((1/(a*b)*np.exp(0+1j*(a+b))*(qz-(delev[0]*qx+delev[1]*qy))))
    
    def scatter(self,graphs = False, results = True,xRange = [-1,1],yRange = [-1,1]):
        '''This is where the main logic of the scattering method is used. graphs is used to check if the user wants graphs
        or not. Results will print out the pressure values at a reciever.'''
        options={'limit':100}
        if self.recieverLocations.shape == (3,): #Making sure we are actually passing co-ordinates. If there is only one reciever then it goes through this statement.
            Real = sp.integrate.nquad(self.Integrand3DReal,[[20*xRange[0],20*xRange[1]],[20*yRange[0],20*yRange[1]]],args =(self.sourceLocation[0],self.sourceLocation[1],self.sourceLocation[2],self.recieverLocations[0],self.recieverLocations[1],self.recieverLocations[2],self.surfaceElevation,self.k),opts = [options,options]) #Numerical Integration
            Imaginary = sp.integrate.nquad(self.Integrand3DImag,[[20*xRange[0],20*xRange[1]],[20*yRange[0],20*yRange[1]]],args =(self.sourceLocation[0],self.sourceLocation[1],self.sourceLocation[2],self.recieverLocations[0],self.recieverLocations[1],self.recieverLocations[2],self.surfaceElevation,self.k),opts = [options,options])
            temp = 1/(4j*np.pi)*( Real[0]+1j*Imaginary[0] ) #The actual answer for KA
            self.Results.append(temp) #Store the values
            self.RealResults.append(temp)
            self.ImaginaryResults.append(temp)
            return self.Results
        else:
            for recieverIndex in range(self.recieverLocations.shape[0]): #Logic for multiple recievers
                Real = sp.integrate.nquad(self.Integrand3DReal,[[20*xRange[0],20*xRange[1]],[20*yRange[0],20*yRange[1]]],args =(self.sourceLocation[0],self.sourceLocation[1],self.sourceLocation[2],self.recieverLocations[recieverIndex][0],self.recieverLocations[recieverIndex][1],self.recieverLocations[recieverIndex][2],self.surfaceElevation,self.k),opts = [options,options])
                Imaginary = sp.integrate.nquad(self.Integrand3DImag,[[20*xRange[0],20*xRange[1]],[20*yRange[0],20*yRange[1]]],args =(self.sourceLocation[0],self.sourceLocation[1],self.sourceLocation[2],self.recieverLocations[recieverIndex][0],self.recieverLocations[recieverIndex][1],self.recieverLocations[recieverIndex][2],self.surfaceElevation,self.k),opts = [options,options])
                temp = 1/(4j*np.pi)*( Real[0]+1j*Imaginary[0] )
                self.Results.append(temp)
                self.RealResults.append(sp.real(temp))
                self.ImaginaryResults.append(sp.imag(temp))
                self.AbsoluteResults.append(np.abs(temp))
                return self.Results
