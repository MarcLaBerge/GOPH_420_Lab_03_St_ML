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

    #Picking the indexes of days 2-5, row number looking at every 24 hours
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

    #Labeling the axis once, not every time
    ax[2].set_ylabel('magnitude [M]')
    ax[4].set_xlabel('time [hr]')
    plt.savefig('figures/data_per_day.png')
    plt.close("all")
    #a, e, rsq = multi_regress(y, Z)


    #Plot the periods of data
    cuts = [34, 46, 72, 96]
    plt.figure(figsize = (15, 15))
    plt.plot(Z,y, 'rx', fillstyle = 'none', label = "Data points")
    plt.xlabel("Time [Hr]", fontsize = 30 )
    plt.ylabel("Magnitude [M]", fontsize = 30)
    plt.grid()

    #Adding lines to better visualise these cuts (vertical)
    for i in cuts:
        plt.vlines(i, -1.5, 1.5, color = 'black', linewidth = 4)
    plt.legend(loc = 'lower left')
    plt.savefig('figures/data_cuts')
    plt.close("all")


    #Set time indices into array and assign the time
    t = 0
    index = np.zeros(0, dtype = int)
    for i in cuts:
        while i > Z[t]:
            t += 1
        index = np.append(index, t)

    #Cuts of the magnitude data during chosen periods
    mag = y[:index[0]]
    mag1 = y[index[0], index[1]]
    mag2 = y[index[1], index[2]]
    mag3 = y[index[2], index[3]]
    mag4 = y[index[3]:]

    #Number of events in each cut
    M = np.linspace(0, 1.5, num = 20)
    N =[sum(1 for j in mag if j > M[m]) for m in range(len(M))] 
    N1 = [sum(1 for j in mag1 if j > M[m]) for m in range(len(M))]
    N2 = [sum(1 for j in mag2 if j > M[m]) for m in range(len(M))]
    N3 = [sum(1 for j in mag3 if j > M[m]) for m in range(len(M))]
    N4 = [sum(1 for j in mag4 if j > M[m]) for m in range(len(M))]
    N_arrays = [N, N1, N2, N3, N4]

    #Create the Z matrix with the info above
    #First row is 1s
    ones = np.ones_like(N)
    #Exponents of equation 4.3, the coefficient b will therefor be +
    Z = np.ones_like(N) * -M 
    #Combining the columns into the final Z matrix
    Z = np.column_stack((ones, Z))
    

if __name__ == "__main__":
    main()


