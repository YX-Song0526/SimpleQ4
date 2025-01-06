from visualize import *
from read_mesh import read_inp

input_file = 'a_b_1.inp'

Node, Element = read_inp(input_file)

n = len(Node)

node_dof_indices = np.arange(0, 2 * n).reshape(-1, 2)

visualize_2d_mesh_plus(Node, Element, node_dof_indices)
