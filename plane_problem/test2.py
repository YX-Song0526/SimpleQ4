import numpy as np
from numpy import pi, cos, sin
from visualize import plot_nodes, visualize_2d_mesh
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 参数
N1 = 9
N2 = 9
R = 4.0
a = 0.3 * R
step = 2 * a / (N1 - 1)

N3 = 4 * N1 - 4
NN = (N1 - 1) * (N1 - 1)

# 角度步长
angle_step = (pi / 2) / (N1 - 1)

square_nodes = np.array(
    [[-a + i * step, a] for i in range(N1 - 1)] +
    [[a, a - i * step] for i in range(N1 - 1)] +
    [[a - i * step, -a] for i in range(N1 - 1)] +
    [[-a, -a + i * step] for i in range(N1 - 1)]
)

circle_nodes = np.array(
    [[-R * cos(pi / 4 + i * angle_step), R * sin(pi / 4 + i * angle_step)] for i in range(4 * (N1 - 1))]
)


# nodes = np.vstack((square_nodes, circle_nodes))


def fill():
    bottom_nodes = square_nodes
    top_nodes = circle_nodes

    inner_nodes = []
    for i in range(1, N2 - 1):
        floor = bottom_nodes + i * (top_nodes - bottom_nodes) / (N2 - 1)
        inner_nodes.extend(floor.tolist())

    return inner_nodes


b = fill()

core = [[-a + i * step, -a + j * step] for j in range(1, N1 - 1) for i in range(1, N1 - 1)]

nodes = np.vstack((square_nodes, b, circle_nodes, core))

#print(len(nodes))
# print(nodes)

# 单元节点索引矩阵，每一行代表每个单元的8个节点索引
outer_elements = np.zeros((N3 * (N2 - 1), 4), dtype=int)
inner_elements = np.zeros((NN, 4), dtype=int)

# 填充单元节点索引矩阵
ie = 0
for j in range(N2 - 1):
    for i in range(N3):
        flag: bool = (i == N3 - 1)
        n1 = j * N3 + i
        n2 = j * N3 if flag else n1 + 1
        n3 = (j + 1) * N3 if flag else n2 + N3
        n4 = n1 + N3

        outer_elements[ie, :] = [n1, n2, n3, n4]
        ie += 1

spy = np.array(
    [[-a + i * step, -a + j * step] for j in range(N1) for i in range(N1)]
)


indices = np.array([np.where(np.isclose(nodes, spy[i], atol=1e-8).all(axis=1))[0][0] for i in range(N1 * N1)])

indices = indices.reshape(N1, N1)

ie = 0
for i in range(N1 - 1):
    for j in range(N1 - 1):
        n1 = indices[i, j]
        n2 = indices[i, j + 1]
        n3 = indices[i + 1, j + 1]
        n4 = indices[i + 1, j]

        inner_elements[ie, :] = [n1, n2, n3, n4]
        ie += 1

elements = np.vstack((outer_elements, inner_elements))

print(elements)

# 找到唯一行
unique_rows, counts = np.unique(elements, axis=0, return_counts=True)

# 检查是否有重复行
if np.any(counts > 1):
    print("存在重复行")
    print(f"重复的行: {unique_rows[counts > 1]}")
else:
    print("没有重复行")

# plot_nodes(nodes)

visualize_2d_mesh(nodes, elements)
