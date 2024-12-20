from stiffness_matrix import *
from preprocess import coord, elements, num_dof, num_nodes, interpolate_stress_to_nodes, node_dof_idx
from visualize import *

K = cal_K_total(coord, elements, num_dof, node_dof_idx)
F = np.zeros(num_dof)
U = np.zeros(num_dof)

F[-2] = -0.02
F[142] = 0.07

K_reduced = K[12:, 12:]

F_reduced = F[12:]
U_reduced = np.linalg.solve(K_reduced, F_reduced)

U[12:] = U_reduced
U1 = U.copy()
U = U.reshape(num_nodes, 2)
coord_new = coord + U
coord_1 = coord_new.copy()

temp = coord_new.copy()
coord_new[:, 0] = temp[:, 1]
coord_new[:, 1] = temp[:, 0]

print(U)

# 可视化
visualize_2d_mesh_plus(coord_new, elements, node_dof_idx)
plot_comparison(coord=coord, coord_new=coord_1, elements=elements, overturn=True)
#
# 计算应力场
element_stresses = calculate_element_stress(U.flatten(), elements, coord, node_dof_idx)
node_stresses = interpolate_stress_to_nodes(elements, element_stresses, num_nodes)

# 绘制变形后的应力云图
plot_stress_cloud(coord, elements, node_stresses, stress_component=1, coord_new=coord_new)
