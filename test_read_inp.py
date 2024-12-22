from stiffness_matrix import cal_K_total, calculate_element_stress
from visualize import *
from read_mesh import read_inp
from preprocess import interpolate_stress_to_nodes


def delete_rows_and_columns(matrix, fixed):
    # 转换输入为 NumPy 数组，以方便操作
    matrix = np.array(matrix)

    # 删除指定行和列
    matrix = np.delete(matrix, fixed, axis=0)  # 删除行
    matrix = np.delete(matrix, fixed, axis=1)  # 删除列

    return matrix


input_file = 'a_b_1.inp'

coord, elements = read_inp(input_file)

num_dof = 2 * len(coord)
num_nodes = len(coord)
node_dof_idx = np.arange(0, num_dof).reshape(-1, 2)

K = cal_K_total(coord, elements, num_dof, node_dof_idx)
F = np.zeros(num_dof)
U = np.zeros(num_dof)

factor = 100000000000
F[20] = 0.5*factor
F[18] = 0.5*factor
F[202] = factor
F[200] = factor
F[198] = factor
F[196] = factor
F[194] = factor
F[192] = factor
F[190] = factor
F[188] = factor
F[186] = factor

fixed_dof = [4, 5, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 6, 7]

K_ff = delete_rows_and_columns(K, fixed=fixed_dof)
F_f = np.delete(F, fixed_dof)

U_f = np.linalg.solve(K_ff, F_f)

# 需要插入的索引对应的是未删除的节点，即非4, 5, 6, 7 索引
remaining_indices = [i for i in range(num_dof) if i not in fixed_dof]

# 将 U_f 的值填充回 U 中
for i, idx in enumerate(remaining_indices):
    U[idx] = U_f[i]

U = U.reshape(num_nodes, 2)
coord_new = coord + U
# print(len(K))
# print(len(coord))
visualize_2d_mesh_plus(coord, elements, node_dof_idx)

plot_comparison(coord=coord, coord_new=coord_new, elements=elements)
#
# 计算应力场
element_stresses = calculate_element_stress(U.flatten(), elements, coord, node_dof_idx)
node_stresses = interpolate_stress_to_nodes(elements, element_stresses, num_nodes)

# 绘制变形后的应力云图
plot_stress_cloud(coord, elements, node_stresses, stress_component=0, coord_new=coord_new)
