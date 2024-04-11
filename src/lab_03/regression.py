import numpy as np



def multi_regress(y, Z):
    """Perform multiple linear regression.

    Parameters
    ----------
    y : array_like, shape = (n,) or (n,1)
        The vector of dependent variable data
    Z : array_like, shape = (n,m)
        The matrix of independent variable data

    Returns
    -------
    a_coef : numpy.ndarray, shape = (m,) or (m,1)
        The vector of model coefficients
    e_res : numpy.ndarray, shape = (n,) or (n,1)
        The vector of residuals
    r_squared : float
        The coefficient of determination, r^2

    """

    #Multiplying Z(transpose) and Z
    ZtZ = np.matmul(np.transpose(Z), Z)

    #Multiplying Z(transpose) and y
    Zty = np.matmul(np.transpose(Z), y)

    #Calculate the residuals
    #Multiply the inverse of ZtZ and Zty
    a_coef = np.matmul(np.linalg.inv(ZtZ), Zty)

    #Calculate the average value of y
    #full_like(Shape, Full value), makes new array with the same shape with specific values
    y_avg = np.full_like(y, np.mean(y))

    #Calculate the residuals
    e_avg = y - y_avg

    #Caluclate the sum of squared residuals
    sy = np.matmul(np.transpose(e_avg), e_avg)

    #Calculate the regression model residuals
    e_res = y - np.matmul(Z, a_coef)

    #Calculate the model's sum of squared residuals
    sr = np.matmul(np.transpose(e_res), e_res)

    #Calculate the R^2 of the model
    r_squared = float((sy - sr) / sy)

    return a_coef, e_res, r_squared









