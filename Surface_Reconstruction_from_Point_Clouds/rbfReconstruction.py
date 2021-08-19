import argparse
import numpy as np
import plotly
import plotly.figure_factory as ff
from skimage import measure
from knnsearch import knnsearch

parser = argparse.ArgumentParser(description='Generate Surface')
parser.add_argument('--file', type=str, default = "data/bunny-500.pts",
                   help='filename', required = False)

def rbfReconstruction(input_point_cloud_filename, epsilon = 1e-7):
    """
    surface reconstruction with an implicit function f(x,y,z) computed
    through RBF interpolation of the input surface points and normals
    input: filename of a point cloud, parameter epsilon
    output: reconstructed mesh
    """

    #load the point cloud
    data = np.loadtxt(input_point_cloud_filename)
    points = data[:,:3]
    normals = data[:,3:]


    # construct a 3D NxNxN grid containing the point cloud
    # each grid point stores the implicit function value
    # set N=16 for quick debugging, use *N=64* for reporting results
    N = 64
    max_dimensions = np.max(points,axis=0) # largest x, largest y, largest z coordinates among all surface points
    min_dimensions = np.min(points,axis=0) # smallest x, smallest y, smallest z coordinates among all surface points
    bounding_box_dimensions = max_dimensions - min_dimensions # compute the bounding box dimensions of the point cloud
    grid_spacing = max(bounding_box_dimensions)/(N-9) # each cell in the grid will have the same size
    X, Y, Z =np.meshgrid(list(np.arange(min_dimensions[0]-grid_spacing*4, max_dimensions[0]+grid_spacing*4, grid_spacing)),
                         list(np.arange(min_dimensions[1] - grid_spacing * 4, max_dimensions[1] + grid_spacing * 4,
                                    grid_spacing)),
                         list(np.arange(min_dimensions[2] - grid_spacing * 4, max_dimensions[2] + grid_spacing * 4,
                                    grid_spacing)))

    IF = np.zeros(shape=X.shape) #this is your implicit function - fill it with correct values!
    # toy implicit function of a sphere - replace this code with the correct
    # implicit function based on your input point cloud!!!
    IF = (X - (max_dimensions[0] + min_dimensions[0]) / 2) ** 2 + \
         (Y - (max_dimensions[1] + min_dimensions[1]) / 2) ** 2 + \
         (Z - (max_dimensions[2] + min_dimensions[2]) / 2) ** 2 - \
         (max(bounding_box_dimensions) / 4) ** 2
    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
    ''' ============================================
    #            YOUR CODE GOES HERE
    ============================================ '''
    points_3n = np.vstack((points, points + epsilon*normals, points - epsilon*normals))
    solution_3n = np.append(np.zeros(points.shape[0]), np.append(np.zeros(points.shape[0])+epsilon,
                                                           np.zeros(points.shape[0])-epsilon))
    print("Points::: {}".format(points.shape))
    print("Points_3N::: {}".format(points_3n.shape))
    print("Solution_3N::: {}".format(solution_3n.shape))

    for i in range(len(points_3n)):
        if i == 0:
            matrix_3n = np.linalg.norm((-1)*points_3n + points_3n[i], axis=1)
        else:
            temp = np.linalg.norm((-1) * points_3n + points_3n[i], axis=1)
            matrix_3n = np.vstack((matrix_3n, temp))
    print("Matrix_3N::: {}".format(np.max(matrix_3n)))
    #print("Matrix_3N::: {}".format(matrix_3n.shape))
    matrix_3n_spline = np.nan_to_num(np.multiply(np.multiply(matrix_3n, matrix_3n), np.log(matrix_3n)))
    print("Matrix_3N_Spline::: {}".format(np.min(matrix_3n_spline)))
    weights = np.linalg.solve(matrix_3n_spline, solution_3n)
    #print("Weights_3N::: {}".format(weights.shape))
    # Calculating the SDF value of grid points

    #Q_3n = np.ones(points_3n.shape[0])
    #Q_3n = np.repeat(Q[:, np.newaxis,:], points_3n.shape[0], axis=1)
    #print("Q_3N::: {}".format(Q_3n.shape))
    #points_new = np.repeat(points_3n[np.newaxis, : , :], Q.shape[0], axis=0)
    #print("points_new::: {}".format(points_new.shape))
    #q_pi = np.subtract(Q_3n,points_new)
    #print("q_pi::: {}".format(q_pi.shape))
    #grid_3n = np.linalg.norm(q_pi, axis=2)
    #grid_3n = np.linalg.norm(Q - points_3n[0], axis=1)
    IF = np.zeros(Q.shape[0])
    for i in range(0,Q.shape[0]):
        r = np.linalg.norm(Q[i] - points_3n, axis=1)
        IF[i] = np.sum(np.multiply(weights, r**2 * np.log(r)))
        #grid_3n = np.append(grid_3n, temp)
    #grid_3n = grid_3n.reshape(( points_3n.shape[0], Q.shape[0])).T
    #print("Grid_3N::: {}".format(grid_3n.shape))
    #grid_3n_spline = np.nan_to_num(np.multiply(np.multiply(grid_3n, grid_3n), np.log(grid_3n)))
    #grid_sol = np.dot(grid_3n_spline, weights)
    #print("GridSol_3N::: {}".format(grid_sol.shape))
    #print("Q::: {}".format(Q.shape))
    IF = IF.reshape(X.shape).T
    #print("IF::: {}".format(IF.shape))

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
    rbfReconstruction(args.file)
