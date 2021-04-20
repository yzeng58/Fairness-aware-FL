import numpy as np
from cvxopt import matrix, solvers
from numpy import genfromtxt
import Adult_dataset
from kernel import gaussian_kernel


def logistic_regression_loss(theta, bias, X, y):
    y_hat = np.dot(X, theta) + bias
    y = y.reshape(-1, 1)
    h = 1/(1 + np.exp(-y_hat))
    loss_y = (-y * np.log(h) - (1 - y) * np.log(1 - h))
    return loss_y, y_hat


def compute_alpha_with_fairness(theta, bias, Xtr, Ytr, Sen, Xref, sigma, B, tau):
    Kt_ref = gaussian_kernel(Xtr, Xref, sigma)

    ##### compute the alpha, we get the linear and quardtic coefficient
    loss_vector, XT_W = logistic_regression_loss(theta, bias, Xtr, Ytr)  ### compute loss of each alpha
    ###linear coefficient
    p_matrix = Kt_ref * loss_vector.reshape(-1, 1)
    p = p_matrix.sum(axis=0)
    p = p.astype('double')

    ### quardtic coefficient
    Q = 2*np.zeros((len(Xref), len(Xref)))
    Q = Q.astype('double')

    ### Compute A, b equality constraint
    A = np.sum(Kt_ref, axis=0).reshape(1, -1)
    b = [1.0*len(Ytr)]
    A = A.astype('double')

    ### compute fariness constraint
    ### compute first term of fairness constraint
    Sen_XT_W = np.multiply(Sen, XT_W)
    fair1_matrix = Kt_ref * Sen_XT_W.reshape(-1, 1)
    fair1 = fair1_matrix.sum(axis=0)
    ### compute second term of fairness constraint
    fair2_matrix = Kt_ref * Sen.reshape(-1, 1)
    fair2 = fair2_matrix.sum(axis=0) / len(Sen)
    distance_sum = XT_W.sum()
    fair2 = fair2 * distance_sum
    fair = (fair1 - fair2).reshape(-1, 1)
    ### G(w, alpha) > tau and G(w, alpha) > -tau
    fair_G = np.concatenate((fair.transpose(), -fair.transpose()))
    fair_h = np.full((2, 1), tau)


    ### compute G and h inequality constraint and combine with fairness inequality
    lower = -np.identity(len(Xref))
    upper = np.identity(len(Xref))
    G = np.concatenate((lower, upper), axis=0)
    G = np.concatenate((G, fair_G))
    lower_h = np.full((len(Xref), 1), 0)
    upper_h = np.full((len(Xref), 1), B)

    h = np.concatenate((lower_h, upper_h), axis=0)
    h = np.concatenate((h, fair_h), axis=0)
    h = h.astype('double')
    Q, p, G, h, A, b = -matrix(Q), -matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)

    sol = solvers.qp(Q, p, G, h, A, b)
    alpha = np.array(sol['x'])
    weight = np.matmul(Kt_ref, alpha)
    ratio = np.sum(weight*Sen) / len(Sen)
    return weight, ratio


def compute_alpha_no_fairness(theta, bias, Xtr, Ytr, Sen, Xref, sigma, B):
    Kt_ref = gaussian_kernel(Xtr, Xref, sigma)

    ##### compute the alpha, we get the linear and quardtic coefficient
    loss_vector, XT_W = logistic_regression_loss(theta, bias, Xtr, Ytr)  ### compute loss of each alpha
    ###linear coefficient
    p_matrix = Kt_ref * loss_vector.reshape(-1, 1)
    p = p_matrix.sum(axis=0)
    p = p.astype('double')

    ### quardtic coefficient
    Q = 2 * np.zeros((len(Xref), len(Xref)))
    Q = Q.astype('double')

    ### Compute A, b equality constraint
    A = np.sum(Kt_ref, axis=0).reshape(1, -1)
    b = [1.0 * len(Ytr)]
    A = A.astype('double')

    ### compute G and h inequality constraint and combine with fairness inequality
    lower = -np.identity(len(Xref))
    upper = np.identity(len(Xref))
    G = np.concatenate((lower, upper), axis=0)
    lower_h = np.full((len(Xref), 1), 0)
    upper_h = np.full((len(Xref), 1), B)

    h = np.concatenate((lower_h, upper_h), axis=0)
    h = h.astype('double')
    Q, p, G, h, A, b = -matrix(Q), -matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)

    sol = solvers.qp(Q, p, G, h, A, b)
    alpha = np.array(sol['x'])
    weight = np.matmul(Kt_ref, alpha)
    ratio = np.sum(weight * Sen) / len(Sen)
    return weight, ratio


def compute_alpha_with_local_fairness(theta, bias, Xtr, Ytr, Sen, Xref, sigma, B, tau, k):
    Kt_ref = gaussian_kernel(Xtr, Xref, sigma)

    ##### compute the alpha, we get the linear and quardtic coefficient
    loss_vector, XT_W = logistic_regression_loss(theta, bias, Xtr, Ytr)  ### compute loss of each alpha
    ###linear coefficient
    p_matrix = Kt_ref * loss_vector.reshape(-1, 1)
    p = p_matrix.sum(axis=0)
    p = p.astype('double')

    ### quardtic coefficient
    Q = 2*np.zeros((len(Xref), len(Xref)))
    Q = Q.astype('double')

    ### Compute A, b equality constraint
    A = np.sum(Kt_ref, axis=0).reshape(1, -1)
    b = [1.0*len(Ytr)]
    A = A.astype('double')

    ### compute fariness constraint
    ### compute first term of fairness constraint

    local_length = len(Sen) // k
    fair1_dict = {}
    fair2_dict = {}
    fair = {}
    Sen_XT_W = np.multiply(Sen, XT_W)
    fair1_matrix = Kt_ref * Sen_XT_W.reshape(-1, 1)

    for i in range(k):
        fair1_dict[i] = fair1_matrix[i*local_length : (i + 1)*local_length].sum(axis=0)
    # fair1 = fair1_matrix.sum(axis=0)

    ### compute second term of fairness constraint
    fair2_matrix = Kt_ref * Sen.reshape(-1, 1)
    for i in range(k):
        temp1 = fair2_matrix[i*local_length : (i + 1)*local_length].sum(axis=0) / local_length
        temp2 = XT_W[i*local_length : (i + 1)*local_length].sum()
        fair2_dict[i] = temp1*temp2


    # fair2 = fair2_matrix.sum(axis=0) / len(Sen)
    # distance_sum = XT_W.sum()
    # fair2 = fair2 * distance_sum
    for i in range(k):
        fair[i] = (fair1_dict[i] - fair2_dict[i]).reshape(-1, 1)


    ### G(w, alpha) > tau and G(w, alpha) > -tau
    fair_G = np.concatenate((fair[0].transpose(), -fair[0].transpose()))
    for i in range(1, k):
        fair_i = np.concatenate((fair[i].transpose(), -fair[i].transpose()))
        fair_G = np.concatenate((fair_G, fair_i))
    fair_h = np.full((2*k, 1), tau)


    ### compute G and h inequality constraint and combine with fairness inequality
    lower = -np.identity(len(Xref))
    upper = np.identity(len(Xref))
    G = np.concatenate((lower, upper), axis=0)
    G = np.concatenate((G, fair_G))
    lower_h = np.full((len(Xref), 1), 0)
    upper_h = np.full((len(Xref), 1), B)

    h = np.concatenate((lower_h, upper_h), axis=0)
    h = np.concatenate((h, fair_h), axis=0)
    h = h.astype('double')
    Q, p, G, h, A, b = -matrix(Q), -matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)

    sol = solvers.qp(Q, p, G, h, A, b)
    alpha = np.array(sol['x'])
    weight = np.matmul(Kt_ref, alpha)
    ratio = np.random.rand(k, 1)
    weight_sen = weight*Sen
    for i in range(k):
        ratio[i]  = np.sum(weight_sen[i*local_length: (i + 1)*local_length])/ local_length

    return weight, ratio

# def logistic_regression_loss(theta, bias, X, y):
#     y_hat = np.dot(X, theta) + bias
#     y = y.reshape(-1, 1)
#     h = 1/(1 + np.exp(-y_hat))
#     loss_y = (-y * np.log(h) - (1 - y) * np.log(1 - h))
#     return loss_y, y_hat

from numpy import genfromtxt

# filename = '/home/wd005/adult_norm.csv'
# numpy_data = genfromtxt(filename, delimiter=',')
#
# index = 30000
# train_data = numpy_data[:, 1:-1][0:index]
# train_label = numpy_data[:, -1][0:index]
# test_data = numpy_data[:, 1:-1][index:]
# test_label = numpy_data[:, -1][index:]
#
# test_data = test_data[0:500]
#
# filename = '/home/wd005/adult_norm.csv'
# adult_data = genfromtxt(filename, delimiter=',')

# length = 40
# train_x = np.random.rand(length, 10)
# train_y = np.random.randint(0, 2, size=(length, 1))
# Sen = np.random.randint(0, 2, size=(length, 1))
# test_ref = np.random.rand(20, 10)
# theta = np.random.rand(10, 1)
# bias = np.random.rand()
# sigma = 1
# B = 5

# def compute_alpha_fairness(theta, bias, Xtr, Ytr, Sen, Xref, sigma, B):
#
#     Kt_ref = gaussian_kernel(Xtr, Xref, sigma)
#
#     ##### compute the alpha, we get the linear and quardtic coefficient
#     loss_vector, XT_W = logistic_regression_loss(theta, bias, Xtr, Ytr) ### compute loss of each alpha
#
#     ###linear coefficient
#     p_matrix = Kt_ref*loss_vector.reshape(-1, 1)
#     p = p_matrix.sum(axis = 0)
#     # print(p.shape)
#     ### quardtic coefficient
#     ### compute first part quardtic coefficient
#     # print(XT_W.shape)
#     # print(Sen.shape)
#     Sen_XT_W = np.multiply(Sen, XT_W)
#     # print(Sen_XT_W.shape)
#     lam1_matrix = Kt_ref*Sen_XT_W.reshape(-1, 1)
#     lam1 = lam1_matrix.sum(axis = 0)
#
#
#     ### compute second part quardtic coefficient
#
#     lam2_matrix = Kt_ref*Sen.reshape(-1, 1)
#
#     lam2 = lam2_matrix.sum(axis = 0) / len(Sen)
#     distance_sum = XT_W.sum()
#     lam2 = lam2*distance_sum
#     lam = (lam1 - lam2).reshape(-1, 1)
#
#     Q = -2*lam*lam.transpose()
#
#
#     # Q = 2 * np.zeros((len(Xref), len(Xref)))
#     # Q = Q.astype('double')
#
#
#     ### Compute A, b
#     A = np.sum(Kt_ref, axis = 0).reshape(1, -1)
#     b = 1.0*len(Ytr)
#     A = A.astype('double')
#
#
#     ### compute G and h
#     lower = -np.identity(len(Xref))
#     upper = np.identity(len(Xref))
#     G = np.concatenate((lower, upper), axis=0)
#     lower_h = np.full((len(Xref), 1), 0)
#     upper_h = np.full((len(Xref), 1), B)
#     h = np.concatenate((lower_h, upper_h), axis=0)
#     h = h.astype('double')
#     # print(np.linalg.matrix_rank([Q; A; G]))
#     Q, p, G, h, A, b = matrix(Q), matrix(p), matrix(G), matrix(h), matrix(A), matrix(b)
#
#     sol = solvers.qp(Q, p, G, h, A, b)
#     alpha = np.array(sol['x'])
#     weight = np.matmul(Kt_ref, alpha)
#     ratio = np.sum(weight*Sen) / len(Sen)
#
#     return weight, ratio




# alpha = compute_alpha_fairness(theta, bias, train_x, train_y, Sen, test_ref, sigma, B)















