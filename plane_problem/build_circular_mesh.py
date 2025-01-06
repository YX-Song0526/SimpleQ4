import numpy as np
from numpy import pi, cos, sin
from visualize import visualize_2d_mesh


def fill_circular_mesh(R, N1, N2):
    """

    Args:
        R:
        N1:
        N2:

    Returns:

    """

    a = 0.3 * R  # 中心正方形边长

    N3 = 4 * N1 - 4
    NN = (N1 - 1) * (N1 - 1)  # 中心单元数

    step = 2 * a / (N1 - 1)  # 长度步长
    angle_step = (pi / 2) / (N1 - 1)  # 角度步长

    # 中心正方形网格点
    square_nodes = np.array(
        [[-a + i * step, a] for i in range(N1 - 1)] +
        [[a, a - i * step] for i in range(N1 - 1)] +
        [[a - i * step, -a] for i in range(N1 - 1)] +
        [[-a, -a + i * step] for i in range(N1 - 1)]
    )

    # 圆形网格点
    circle_nodes = np.array(
        [[-R * cos(pi / 4 + i * angle_step), R * sin(pi / 4 + i * angle_step)] for i in range(4 * (N1 - 1))]
    )

    # 填充正方形和圆形之间的网格点
    filling = []
    for i in range(1, N2 - 1):
        floor = square_nodes + i * (circle_nodes - square_nodes) / (N2 - 1)
        filling.extend(floor.tolist())

    # 正方形内的网格点
    core = [[-a + i * step, -a + j * step] for j in range(1, N1 - 1) for i in range(1, N1 - 1)]

    # 整合数据
    Node = np.vstack((square_nodes, filling, circle_nodes, core))

    # 创建单元节点索引矩阵，分外部单元和内部单元
    outer_elements = np.zeros((N3 * (N2 - 1), 4), dtype=int)
    inner_elements = np.zeros((NN, 4), dtype=int)

    # 填充外部单元节点索引矩阵
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

    # 试探节点
    spy = np.array(
        [[-a + i * step, -a + j * step] for j in range(N1) for i in range(N1)]
    )

    # 中心正方形节点索引
    indices = np.array(
        [np.where(np.isclose(Node, spy[i], atol=1e-8).all(axis=1))[0][0] for i in range(N1 * N1)]
    )

    # 将索引转为二维数组
    indices = indices.reshape(N1, N1)

    # 填充内部单元节点索引矩阵
    ie = 0
    for i in range(N1 - 1):
        for j in range(N1 - 1):
            n1 = indices[i, j]
            n2 = indices[i, j + 1]
            n3 = indices[i + 1, j + 1]
            n4 = indices[i + 1, j]

            inner_elements[ie, :] = [n1, n2, n3, n4]
            ie += 1

    # 整合单元数据
    Element = np.vstack((outer_elements, inner_elements))

    return Node, Element


nodes, elements = fill_circular_mesh(4, 9, 9)

print(len(nodes), len(elements))

visualize_2d_mesh(nodes, elements)
