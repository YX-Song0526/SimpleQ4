import numpy as np

# 材料属性
E = 69e9  # 杨氏模量
niu = 0.28  # 泊松比
rho = 1  # 单位密度

# 弹性矩阵
D = (E / (1 - niu ** 2)) * np.array([[1, niu, 0],
                                     [niu, 1, 0],
                                     [0, 0, (1 - niu) / 2]])

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


def cal_Ke(node_coords):
    """
    计算单元刚度矩阵

    Args:
        node_coords: 节点坐标矩阵 (num_nodes, 2)
    """
    Ke = np.zeros((8, 8))  # 修改大小以适应 2D 单元
    for gp_t, wt in zip(gauss_points, gauss_weights):
        for gp_s, ws in zip(gauss_points, gauss_weights):
            # 计算形函数导数
            dN_dxi = shape_func_derivatives(gp_s, gp_t)

            # 雅可比矩阵计算
            J = dN_dxi.T @ node_coords
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError("Jacobian determinant is non-positive!")
            J_inv = np.linalg.inv(J)

            # B 矩阵计算
            dN_dxy = J_inv @ dN_dxi.T
            B = np.zeros((3, 8))
            B[0, 0::2] = dN_dxy[0, :]  # 输出索引是 [0, 2, 4, 6]
            B[1, 1::2] = dN_dxy[1, :]  # 输出索引是 [1, 3, 5, 7]
            B[2, 0::2] = dN_dxy[1, :]
            B[2, 1::2] = dN_dxy[0, :]

            # 刚度矩阵累加
            Ke += B.T @ D @ B * detJ * wt * ws
    return Ke


def cal_K_total(node_coords, elements, num_dof, node_dof_idx):
    """
    计算系统总体刚度矩阵

    Args:
        node_dof_idx: 节点自由度索引矩阵
        num_dof: 自由度数目
        elements: 单元连接矩阵 (num_elements, 4)
        node_coords: 节点坐标矩阵 (num_nodes, 2)
    """
    K_total = np.zeros((num_dof, num_dof))
    for element in elements:
        ele_nodes = node_coords[element]
        Ke = cal_Ke(ele_nodes)
        dof = node_dof_idx[element].reshape(8)
        for i in range(8):
            for j in range(8):
                K_total[dof[i], dof[j]] += Ke[i, j]

    return K_total


def cal_Me(node_coords):
    """
    计算单元质量质量矩阵

    Args:
        node_coords: 节点坐标矩阵 (num_nodes, 2)
    """
    Me = np.zeros((8, 8))  # 修改大小以适应 2D 单元
    for gp_t, wt in zip(gauss_points, gauss_weights):
        for gp_s, ws in zip(gauss_points, gauss_weights):
            # 计算形函数
            N = shape_func(gp_s, gp_t)
            N_matrix = np.zeros((2, 8))
            N_matrix[0, 0::2] = N
            N_matrix[1, 1::2] = N

            # 雅可比矩阵计算
            dN_dxi = shape_func_derivatives(gp_s, gp_t)
            J = dN_dxi.T @ node_coords
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError("Jacobian determinant is non-positive!")

            # 质量矩阵累加
            Me += rho * (N_matrix.T @ N_matrix) * detJ * wt * ws
    return Me


def cal_M_total(node_coords, elements, num_dof, node_dof_idx):
    """
    计算系统总体质量矩阵

    Args:
        node_dof_idx: 节点自由度索引矩阵
        num_dof: 自由度数目
        elements: 单元连接矩阵 (num_elements, 4)
        node_coords: 节点坐标矩阵 (num_nodes, 2)
    """
    M_total = np.zeros((num_dof, num_dof))
    for element in elements:
        ele_nodes = node_coords[element]
        Me = cal_Me(ele_nodes)
        dof = node_dof_idx[element].reshape(8)
        # print(dof)
        for i in range(8):
            for j in range(8):
                M_total[dof[i], dof[j]] += Me[i, j]

    return M_total


def calculate_element_stress(U, elements, node_coords, node_dof_idx):
    """
    计算单元应力。

    Args:
        node_dof_idx: 节点自由度索引矩阵
        U: 节点位移向量
        elements: 单元连接矩阵 (num_elements, 4)
        node_coords: 节点坐标矩阵 (num_nodes, 2)
    """
    element_stresses = []
    for element in elements:
        ele_nodes = node_coords[element]
        ele_dof = node_dof_idx[element].reshape(-1)
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
