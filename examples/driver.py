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
    magnitude = data[:,1]

    #Independant data -> time
    time = data[:,0]

    #Picking the indexes of days 2-5, row number looking at every 24 hours
    days = [3928, 5684, 6974, 9536]

    #Plotting the data for each day
    fig, ax = plt.subplots(5, figsize = (15,15))
    #Day one plot
    ax[0].plot(time[:days[0]], magnitude[:days[0]], 'go', fillstyle = 'none', label = 'Day 1 plot')
    #Day two plot
    ax[1].plot(time[days[0]:days[1]], magnitude[days[0]:days[1]], 'go', fillstyle = 'none', label = 'Day 2 plot')
    #Day three plot
    ax[2].plot(time[days[1]:days[2]], magnitude[days[1]:days[2]], 'go', fillstyle = 'none', label = 'Day 3 plot')
    #Day four plot 
    ax[3].plot(time[days[2]:days[3]], magnitude[days[2]:days[3]], 'go', fillstyle = 'none', label = 'Day 4 plot')
    #Day five plot 
    ax[4].plot(time[days[3]:], magnitude[days[3]:], 'go', fillstyle = 'none', label = 'Day 4 plot')
    for i in ax:
        i.legend(loc = "upper right")

    #Labeling the axis once, not every time
    ax[2].set_ylabel('Magnitude [M]', fontsize = 25)
    ax[4].set_xlabel('Time [hr]', fontsize = 25)
    plt.savefig('figures/data_per_day.png')
    plt.close("all")
    #a, e, rsq = multi_regress(y, Z)


    #Plot the periods of data
    cuts = [34, 46, 72, 96]
    plt.figure(figsize = (15, 15))
    plt.plot(time,magnitude, 'rx', fillstyle = 'none', label = "Data points")
    plt.xlabel("Time [Hr]", fontsize = 25)
    plt.ylabel("Magnitude [M]", fontsize = 25)
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
        while i > time[t]:
            t += 1
        index = np.append(index, t)

    #Cuts of the magnitude data during chosen periods
    y0 = magnitude[:index[0]]
    y1 = magnitude[index[0]: index[1]]
    y2 = magnitude[index[1]: index[2]]
    y3 = magnitude[index[2]: index[3]]
    y4 = magnitude[index[3]:]

    #Number of events in each cut
    #interval has to not divide by 0, and where linear regression works
    M = np.linspace(-0.25, 1, num = 25)

    #Counting the number of values with magnitudes greater than M(k), interval limit is the same as M
    N0 = [sum(1 for j in y0 if j > M[k]) for k in range(len(M))]
    N1 = [sum(1 for j in y1 if j > M[k]) for k in range(len(M))]
    N2 = [sum(1 for j in y2 if j > M[k]) for k in range(len(M))]
    N3 = [sum(1 for j in y3 if j > M[k]) for k in range(len(M))]
    N4 = [sum(1 for j in y4 if j > M[k]) for k in range(len(M))]
    N_arrays = [N0, N1, N2, N3, N4]

    #Create the Z matrix with the info above
    #First row is 1s
    ones = np.ones_like(N0)
    #Exponents of equation 4.3, the coefficient b will therefor be +
    Z = np.ones_like(N0) * -M 
    #Combining the columns into the final Z matrix
    Z = np.column_stack((ones, Z))

    #Creating an array of arrays for the residuals
    r0 = np.zeros(0)
    r1 = np.zeros(0)
    r2 = np.zeros(0)
    r3 = np.zeros(0)
    r4 = np.zeros(0)
    res = [r0,r1,r2,r3,r4]

    #Creating an array of labels to make it easier to name
    labels = ['Period_1', 'Period_2', 'Period_3', 'Period_4', 'Period_5']
    
    for j, num in enumerate(N_arrays):
        #Keeping in mind that to solve equation 4.3, the log will have to be taken
        plt.figure()
        a, e, rsq = multi_regress(np.log10(num), Z)
        rgr = 10 ** np.matmul(Z, a)
        res[j] = np.append(res[j], e)
        #Plotting a log scale on the y axis
        plt.semilogx(num, M, 'bx', label = 'Number of events with \ncorresponding magnitudes'
                     +f'$\geq$M')
        #labeling 4.3 in figure, with decimal points
        equation1 = f"10^({a[0]:.2f} - {a[1]:.2f}M) \n $R^2$ = {rsq:.4f}"
        #equation2 = f"$R^2$ = {rsq:.4f}"
        #Plotting the model
        plt.plot(rgr, M, '.-.r', label = equation1) #and equation2)
        plt.ylabel("Magnitudes [M]")
        plt.xlabel("Number of events")
        plt.legend()
        plt.savefig(f"figures/{labels[j]}.png")

if __name__ == "__main__":
    main()


