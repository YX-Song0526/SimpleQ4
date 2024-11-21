import numpy as np

# 节点数
Nx = 6
Ny = 26
# Nz = 2

num_nodes = Nx * Ny  # 节点总数
num_elements = (Nx - 1) * (Ny - 1)  # 单元总数

x_start, x_end = -1.5, 1.5
y_start, y_end = 0, 30


x_step, y_step = (x_end - x_start) / (Nx - 1), (y_end - y_start) / (Ny - 1)

# 填充坐标矩阵
coord = np.array([[x_start + i * x_step, y_start + j * y_step]
                  for j in range(Ny) for i in range(Nx)])

# 单元节点索引矩阵，每一行代表每个单元的8个节点索引
elements = np.zeros((num_elements, 4), dtype=int)

# 填充单元节点索引矩阵
ie = 0
for j in range(Ny - 1):
    for i in range(Nx - 1):
        n1 = j * Nx + i
        n2 = n1 + 1
        n3 = n2 + Nx
        n4 = n3 - 1

        elements[ie, :] = [n1, n2, n3, n4]
        ie += 1

# 节点自由度索引矩阵，每一行代表每个节点的2个自由度索引
node_dof_idx = np.zeros((num_nodes, 2), dtype=int)

# 填充自由度索引矩阵
n = 0
for i in range(num_nodes):
    node_dof_idx[i, :] = [n, n + 1]
    n += 2

# 自由度总数
num_dof = n


def interpolate_stress_to_nodes(elements, element_stresses, num_nodes):
    node_stresses = np.zeros((num_nodes, 3))  # 节点应力
    counts = np.zeros(num_nodes)  # 每个节点被访问的次数

    for i, element in enumerate(elements):
        for node in element:
            node_stresses[node] += element_stresses[i]
            counts[node] += 1

    # 归一化，求平均
    node_stresses /= counts[:, None]
    return node_stresses
