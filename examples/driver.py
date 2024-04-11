import numpy as np
import matplotlib as plt
from lab_03.regression import multi_regress


def main():
    """
    Driver script implementation of multi linear regression for earthquake magnitude data.
    """
    #Grabbing data from files on the repository
    data = np.loadtxt('data/M_data.txt')

    #Dependant data 
    y = ... # enter dependent variable data (or load from a file)

    #Independant data 
    Z = ... # enter independent variable data (or load from a file)
    a, e, rsq = multi_regress(y, Z)

if __name__ == "__main__":
    main()


