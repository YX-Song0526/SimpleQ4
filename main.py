import numpy as np
from stiffness_matrix import *
from preprocess import coord, elements, num_dof, num_nodes, interpolate_stress_to_nodes
from visualize import *

K = cal_K_total(coord, elements)
F = np.zeros(num_dof)
U = np.zeros(num_dof)

F[-11::2] = -1
F[-1] = -0.5
F[-11] = -0.5

K_reduced = K[12:, 12:]

F_reduced = F[12:]
U_reduced = np.linalg.solve(K_reduced, F_reduced)

U[12:] = U_reduced
U = U.reshape(num_nodes, 2)
coord_new = coord + U

print(U)

# 可视化
# visualize_2d_mesh(coord_new, elements)
plot_comparison(coord, coord_new, elements)


# 计算应力场
element_stresses = calculate_element_stress(U.flatten(), elements, coord)
node_stresses = interpolate_stress_to_nodes(elements, element_stresses, num_nodes)

# 绘制变形后的应力云图
plot_stress_cloud(coord, elements, node_stresses, stress_component=0, coord_new=coord_new)
