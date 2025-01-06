import numpy as np

# 定义高斯积分点和权重
gauss_points = [-np.sqrt(1 / 3), np.sqrt(1 / 3)]
gauss_weights = [1, 1]


# 定义形函数
def shape_func(s, t):
    N = np.array([(1 - s) * (1 - t),
                  (1 + s) * (1 - t),
                  (1 + s) * (1 + t),
                  (1 - s) * (1 + t)]) / 4
    return N


# 定义形函数的导数矩阵
def shape_func_derivatives(s, t):
    dN_dxi = np.array([[-(1 - t), -(1 - s)],
                       [(1 - t), -(1 + s)],
                       [(1 + t), (1 + s)],
                       [-(1 + t), (1 - s)]]) / 4
    return dN_dxi


def calculate_element_stress(U,
                             Element,
                             Node,
                             node_dof_indices,
                             D):
    """
    计算单元应力。

    Args:
        D: 弹性矩阵
        node_dof_indices: 节点自由度索引矩阵
        U: 节点位移向量
        Element: 单元连接矩阵 (num_elements, 4)
        Node: 节点坐标矩阵 (num_nodes, 2)
    """
    element_stresses = []
    for element in Element:
        ele_nodes = Node[element]
        ele_dof = node_dof_indices[element].reshape(-1)
        U_e = U[ele_dof]

        # 初始化高斯点应力累加
        stress_at_gauss_points = []

        for gp_t, wt in zip(gauss_points, gauss_weights):
            for gp_s, ws in zip(gauss_points, gauss_weights):
                # 计算形函数导数
                dN_dxi = shape_func_derivatives(gp_s, gp_t)
                J = dN_dxi.T @ ele_nodes
                J_inv = np.linalg.inv(J)
                dN_dxy = J_inv @ dN_dxi.T

                # 计算 B 矩阵
                B = np.zeros((3, 8))
                B[0, 0::2] = dN_dxy[0, :]
                B[1, 1::2] = dN_dxy[1, :]
                B[2, 0::2] = dN_dxy[1, :]
                B[2, 1::2] = dN_dxy[0, :]

                # 计算应变和应力
                strain = B @ U_e
                stress = D @ strain
                stress_at_gauss_points.append(stress)

        # 对所有高斯点的应力取平均
        element_stresses.append(np.mean(stress_at_gauss_points, axis=0))

    return np.array(element_stresses)


def interpolate_stress_to_nodes(Element,
                                element_stresses,
                                num_nodes):
    node_stresses = np.zeros((num_nodes, 3))  # 节点应力
    counts = np.zeros(num_nodes)  # 每个节点被访问的次数

    for i, element in enumerate(Element):
        for node in element:
            node_stresses[node] += element_stresses[i]
            counts[node] += 1

    # 归一化，求平均
    node_stresses /= counts[:, None]
    return node_stresses
