import numpy as np
import sklearn.neighbors
import scipy.sparse
import warnings
import os
import scipy

nn = 10

"""
    taken from https://github.com/MarcoForte/knn-matting/blob/master/donkeyTrimap.png
    get data from: http://alphamatting.com/datasets.php
"""

def knn_matte(img, trimap, mylambda=100):
    [m, n, c] = img.shape
    img, trimap = img/255.0, trimap/255.0
    foreground = (trimap > 0.99).astype(int)
    background = (trimap < 0.01).astype(int)
    all_constraints = foreground + background

    print('Finding nearest neighbors')
    a, b = np.unravel_index(np.arange(m*n), (m, n))
    feature_vec = np.append(np.transpose(img.reshape(m*n,c)), [ a, b]/np.sqrt(m*m + n*n), axis=0).T
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=10, n_jobs=4).fit(feature_vec)
    knns = nbrs.kneighbors(feature_vec)[1]

    # Compute Sparse A
    print('Computing sparse A')
    row_inds = np.repeat(np.arange(m*n), 10)
    col_inds = knns.reshape(m*n*10)
    vals = 1 - np.linalg.norm(feature_vec[row_inds] - feature_vec[col_inds], axis=1)/(c+2)
    A = scipy.sparse.coo_matrix((vals, (row_inds, col_inds)),shape=(m*n, m*n))

    D_script = scipy.sparse.diags(np.ravel(A.sum(axis=1)))
    L = D_script-A
    D = scipy.sparse.diags(np.ravel(all_constraints[:,:, 0]))
    v = np.ravel(foreground[:,:,0])
    c = 2*mylambda*np.transpose(v)
    H = 2*(L + mylambda*D)

    print('Solving linear system for alpha')
    warnings.filterwarnings('error')
    alpha = []
    try:
        alpha = np.minimum(np.maximum(scipy.sparse.linalg.spsolve(H, c), 0), 1).reshape(m, n)
    except Warning:
        x = scipy.sparse.linalg.lsqr(H, c)
        alpha = np.minimum(np.maximum(x[0], 0), 1).reshape(m, n)
    return alpha

"""
    refine KNN-matting results using data from 
"""
def get_images_for_fcn(num_images, s):
    # get num_images images form the path and put as a matrix
    imgs = []
    num = 0
    path = '/Users/yu-chieh/Downloads/input_training_lowres/'
    for f in os.listdir(path)[s:]:
        if num >= num_images:
            return np.array(imgs)
        image_path = os.path.join(path,f)
        image = scipy.misc.imread(image_path, mode='RGB')
        imgs.append(image)
        num += 1
        print(f)
    return np.array(imgs)

def get_trimap_for_fcn(num_images, s):
    # get num_images images form the path and put as a matrix
    imgs = []
    num = 0
    path = '/Users/yu-chieh/Downloads/trimap_training_lowres/Trimap1'
    for f in os.listdir(path)[s:]:
        if num >= num_images:
            return np.array(imgs)
        image_path = os.path.join(path,f)
        image = scipy.misc.imread(image_path, mode='RGB')
        # cheating version
        # image = np.dstack((image, get_xy_mask(image)))
        imgs.append(image)
        num += 1
        print(f)
    return np.array(imgs)

def get_filenames(num_images, s):
    path = '/Users/yu-chieh/Downloads/input_training_lowres/'
    return os.listdir(path)[s:]


def save_knn_mattes(imgs, trimaps, filenames, mylambda=100):
    for i, t, f in zip(imgs, trimaps, filenames):
        alpha = knn_matte(i, t)
        scipy.misc.imsave('knn_alpha/' + f + '.png', alpha)


def main():
    amount = 27
    filenames = get_filenames(amount, 10)
    imgs = get_images_for_fcn(amount, 10)
    trimaps = get_trimap_for_fcn(amount, 10)
    save_knn_mattes(imgs, trimaps, filenames, mylambda=100)

    # img = scipy.misc.imread('donkey.png')[:,:,:3]
    # trimap = scipy.misc.imread('donkeyTrimap.png')[:,:,:3]
    # alpha = knn_matte(img, trimap)
    # scipy.misc.imsave('donkeyAlpha.png', alpha)
    # plt.title('Alpha Matte')
    # plt.imshow(alpha, cmap='gray')
    # plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy.misc
    main()