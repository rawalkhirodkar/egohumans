import numpy as np
import random
from utils.transforms import linear_transform

# https://github.com/nkolot/SPIN/blob/5c796852ca7ca7373e104e8489aa5864323fbf84/utils/pose_utils.py#L60
def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat, scale, R, t

### aria (S1), colmap(S2). map aria to colmap
def inner_procrustes_alignment(S1, S2):

    S1 = S1.T ## N x 3 -> 3 x N
    S2 = S2.T ## N x 3 -> 3 x N

    S1_hat, scale, R, t = compute_similarity_transform(S1, S2)

    l2_error = ((S1_hat - S2)**2).mean()

    ## make T out of the scale, R and t
    T = np.eye(4)
    T[:3, :3] = scale*R
    # T[:3, :3] = R

    T[:3, 3] = t.reshape(-1)

    output = {'scale': scale, 'R': R, 't': t}

    return T, l2_error, output

### aria (S1), colmap(S2). map aria to colmap
### converts S1 to S2
def procrustes_alignment(S1, S2, num_iters=500, alignment_l2_error_epsilon=1e-3):
    assert(len(S1) == len(S2))
    timestamps_set = set(range(len(S1)))
    inlier_set = set()

    for i in range(num_iters):
        sampled_timestamps = sorted(random.sample(timestamps_set, 3))

        S1_mini = S1[sampled_timestamps].copy()
        S2_mini = S2[sampled_timestamps].copy()

        T_mini, l2_error_mini, output_mini =  inner_procrustes_alignment(S1_mini, S2_mini)
        new_inlier_set = set(sampled_timestamps)
        S2_hat = linear_transform(points_3d=S1, T=T_mini)

        for timestamp in timestamps_set:
            alignment_l2_error = ((S2_hat[timestamp] - S2[timestamp])**2).mean()

            if alignment_l2_error < alignment_l2_error_epsilon:
                new_inlier_set.add(timestamp)


        if len(new_inlier_set) > len(inlier_set):
            inlier_set = new_inlier_set


    if len(inlier_set) == 0:
        inlier_set = timestamps_set.copy()

    S1_inlier = S1[sorted(list(inlier_set))].copy()
    S2_inlier = S2[sorted(list(inlier_set))].copy()

    T, l2_error, output =  inner_procrustes_alignment(S1_inlier, S2_inlier)

    return T, l2_error, output