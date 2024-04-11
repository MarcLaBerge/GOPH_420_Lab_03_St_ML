import numpy as np
import matplotlib.pyplot as plt
from lab_03.regression import multi_regress


def main():
    """
    Driver script implementation of multi linear regression for earthquake magnitude data.
    """
    #Grabbing data from files on the repository
    data = np.loadtxt('data/M_data.txt')

    #Dependant data -> Magnitude
    y = data[:,1]

    #Independant data -> time
    Z = data[:,0]

    #Picking the indexes of days 2-5
    days = [3928, 5684, 6974, 9536]

    #Plotting the data for each day
    fig, ax = plt.subplots(5, figsize = (15,15))
    #Day one plot
    ax[0].plot(Z[:days[0]], y[:days[0]], 'go', fillstyle = 'none', label = 'Day 1 plot')
    #Day two plot
    ax[1].plot(Z[days[0]:days[1]], y[days[0]:days[1]], 'go', fillstyle = 'none', label = 'Day 2 plot')
    #Day three plot
    ax[2].plot(Z[days[1]:days[2]], y[days[1]:days[2]], 'go', fillstyle = 'none', label = 'Day 3 plot')
    #Day four plot 
    ax[3].plot(Z[days[2]:days[3]], y[days[2]:days[3]], 'go', fillstyle = 'none', label = 'Day 4 plot')
    #Day five plot 
    ax[4].plot(Z[days[3]:], y[days[3]:], 'go', fillstyle = 'none', label = 'Day 4 plot')
    for i in ax:
        i.legend(loc = "upper right")

    ax[2].set_ylabel('magnitude [M]')
    ax[4].set_xlabel('time [hr]')
    plt.savefig('figures/data_per_day.png')
    #a, e, rsq = multi_regress(y, Z)

if __name__ == "__main__":
    main()


