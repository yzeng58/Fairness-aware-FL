import numpy as np
def gaussian_kernel(X1, X2, sigma):
    t1= X1.shape[0]
    t2 = X2.shape[0]

    diag1 = np.diag(np.matmul(X1, X1.transpose())).reshape(t1, 1)
    diag2 = np.diag(np.matmul(X2, X2.transpose())).transpose().reshape(1, t2)

    D1 = np.tile(diag1, t2)
    D2 = np.tile(diag2, t1).reshape(D1.shape[0], D1.shape[1])
    D3 = 2*np.matmul(X1, X2.transpose())
    D = D1 + D2 - D3
    K = np.exp(-0.5*D/sigma)

    return K




