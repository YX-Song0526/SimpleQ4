import numpy as np
from scipy.linalg import eigh
from stiffness_matrix import cal_K_total, cal_M_total, calculate_element_stress
from preprocess import coord, elements, num_nodes, num_dof, interpolate_stress_to_nodes, node_dof_idx
from visualize import plot_stress_cloud

# 计算全局刚度矩阵和全局质量矩阵
K = cal_K_total(coord, elements, num_dof, node_dof_idx)
M = cal_M_total(coord, elements, num_dof, node_dof_idx)

# 两端固定边界条件
K_reduced = K[12:-12, 12:-12]
M_reduced = M[12:-12, 12:-12]

# 求解广义特征值问题
eigenvalues, eigenvectors = eigh(K_reduced, M_reduced)

# 计算模态频率（以Hz表示）
frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

U = np.zeros(num_dof)
U[12:-12] = eigenvectors[:, 2]  # 第3阶振型
U = U.reshape(num_nodes, 2)

coord_new = coord + 10 * U  # 将位移放大十倍
print(U)

# 计算应力场
element_stresses = calculate_element_stress(U.flatten(), elements, coord, node_dof_idx)
node_stresses = interpolate_stress_to_nodes(elements, element_stresses, num_nodes)

# 绘制变形后的应力云图
plot_stress_cloud(coord, elements, node_stresses, stress_component=1, coord_new=coord_new)
