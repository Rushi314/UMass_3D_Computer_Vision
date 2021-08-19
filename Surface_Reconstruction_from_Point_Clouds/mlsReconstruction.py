import argparse
import numpy as np
import plotly
import plotly.figure_factory as ff
from skimage import measure
from knnsearch import knnsearch

parser = argparse.ArgumentParser(description='Generate Surface')
parser.add_argument('--file', type=str, default="data/sphere.pts",
                    help='filename', required=False)


def mlsReconstruction(input_point_cloud_filename):
    """
    surface reconstruction with an implicit function f(x,y,z) representing
    MLS distance to the tangent plane of the input surface points
    input: filename of a point cloud
    output: reconstructed mesh
    """

    # load the point cloud
    data = np.loadtxt(input_point_cloud_filename)
    points = data[:, :3]
    normals = data[:, 3:]

    # construct a 3D NxNxN grid containing the point cloud
    # each grid point stores the implicit function value
    # set N=16 for quick debugging, use *N=64* for reporting results
    N = 64
    max_dimensions = np.max(points, axis=0)  # largest x, largest y, largest z coordinates among all surface points
    min_dimensions = np.min(points, axis=0)  # smallest x, smallest y, smallest z coordinates among all surface points
    bounding_box_dimensions = max_dimensions - min_dimensions  # compute the bounding box dimensions of the point cloud
    grid_spacing = max(bounding_box_dimensions) / (N - 9)  # each cell in the grid will have the same size
    X, Y, Z = np.meshgrid(
        list(np.arange(min_dimensions[0] - grid_spacing * 4, max_dimensions[0] + grid_spacing * 4, grid_spacing)),
        list(np.arange(min_dimensions[1] - grid_spacing * 4, max_dimensions[1] + grid_spacing * 4,
                       grid_spacing)),
        list(np.arange(min_dimensions[2] - grid_spacing * 4, max_dimensions[2] + grid_spacing * 4,
                       grid_spacing)))

    IF = np.zeros(shape=X.shape)
    # toy implicit function of a sphere - replace this code with the correct
    # implicit function based on your input point cloud!!!
    IF = (X - (max_dimensions[0] + min_dimensions[0]) / 2) ** 2 + \
         (Y - (max_dimensions[1] + min_dimensions[1]) / 2) ** 2 + \
         (Z - (max_dimensions[2] + min_dimensions[2]) / 2) ** 2 - \
         (max(bounding_box_dimensions) / 4) ** 2

    # idx stores the index to the nearest surface point for each grid point.
    # we use provided knnsearch function
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    R = points
    K = 20
    idx = knnsearch(Q, R, K)
    nearest_points = R[idx]

    # Calculating beta
    idx_beta = knnsearch(R, R, 1)
    beta = 2*np.mean(np.linalg.norm(R-R[idx_beta].reshape(R.shape), axis=1))
    print(beta)

    ''' ============================================
    #            YOUR CODE GOES HERE
    ============================================ '''

    p_pi = -nearest_points + Q.reshape((nearest_points.shape[0],1, 3))
    exp_p_pi = np.exp((-1*np.linalg.norm(p_pi, axis=2)**2)/(beta**2))
    print(exp_p_pi.shape)
    numerator = np.sum(np.multiply(np.sum(np.multiply(normals[idx],p_pi), 2),exp_p_pi), 1)
    denominator = np.sum(exp_p_pi, 1)
    IF = numerator/denominator
    IF = IF.reshape(X.shape).T

    ''' ============================================
    #              END OF YOUR CODE
    ============================================ '''

    verts, simplices = measure.marching_cubes_classic(IF, 0)

    x, y, z = zip(*verts)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=simplices,
                            title="Isosurface")
    plotly.offline.plot(fig)


if __name__ == '__main__':
    args = parser.parse_args()
    mlsReconstruction(args.file)

