import numpy as np
from parameters import *
from read_mesh import read_inp
from shape_function import gauss_points, gauss_weights, shape_func_derivatives, calculate_element_stress, \
    interpolate_stress_to_nodes
from typing import Optional
from visualize import visualize_2d_mesh_plus, plot_stress_cloud
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class PlaneElastic:
    def __init__(self, problem_type: str = PLANE_STRESS):
        self.problem_type: str = problem_type
        self.Node: Optional[np.ndarray] = None
        self.Element: Optional[np.ndarray] = None
        self.E: float = 0.0
        self.niu: float = 0.0
        self.D: Optional[np.ndarray] = None
        self.num_nodes: int = 0
        self.num_dof: int = 0
        self.node_dof_indices: Optional[np.ndarray] = None
        self.sets: dict[str, list[int]] = {}
        self.fixed_dof = []
        self.K: Optional[np.ndarray] = None
        self.U: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.stress = None

    def add_nodes(self, nodes: np.ndarray):
        """
        添加节点数据

        Args:
            nodes: 节点坐标矩阵，形状为 (num_nodes, 2)

        """
        self.Node = nodes

        n = len(nodes)  # 获取节点数
        self.num_nodes = n
        self.num_dof = 2 * n  # 总自由度数
        self.F = np.zeros(2 * n)  # 初始化节点力向量
        self.U = np.zeros(2 * n)  # 初始化节点位移矩阵
        self.node_dof_indices = np.arange(0, 2 * n).reshape(-1, 2)  # 节点自由度索引

    def add_elements(self, elements: np.ndarray):
        """
        添加单元节点编号数据

        Args:
            elements: 单元节点编号矩阵，形状为 (num_elements, 4)

        """
        self.Element = elements

    def load_mesh(self, file_path: str):
        """
        从 .inp 文件中导入网格

        Args:
            file_path: .inp 文件路径

        """
        nodes, elements = read_inp(file_path, 2)
        self.add_nodes(nodes)
        self.add_elements(elements)

    def set_material(self, E, niu):
        """
        设置材料参数

        Args:
            E: 杨氏模量
            niu: 泊松比

        """
        self.E = E
        self.niu = niu

        if self.problem_type == PLANE_STRESS:
            # 平面应力问题的弹性矩阵
            self.D = (E / (1 - niu ** 2)) * np.array([[1, niu, 0],
                                                      [niu, 1, 0],
                                                      [0, 0, (1 - niu) / 2]])

        elif self.problem_type == PLANE_STRAIN:
            # 平面应变问题的弹性矩阵
            self.D = (E / ((1 - 2 * niu) * (1 + niu))) * np.array([[1 - niu ** 2, niu, 0],
                                                                   [niu, 1 - niu ** 2, 0],
                                                                   [0, 0, (1 - 2 * niu) / 2]])

    def cal_K_total(self):
        """计算总体刚度矩阵"""

        def cal_Ke(node_coords):
            """计算单元刚度矩阵"""
            K_e = np.zeros((8, 8))  # 修改大小以适应 2D 单元
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
                    # K_e += B.T @ self.D @ B * detJ * wt * ws
                    # 添加轴对称系数 2πr 计算
                    r = np.mean(node_coords[:, 0])  # 计算单元的平均半径
                    K_e += B.T @ self.D @ B * detJ * wt * ws * 2 * np.pi * r
            return K_e

        n = self.num_dof

        K_total = np.zeros((n, n))
        for element in self.Element:
            ele_nodes = self.Node[element]
            Ke = cal_Ke(ele_nodes)
            dof = self.node_dof_indices[element].reshape(8)
            for i in range(8):
                for j in range(8):
                    K_total[dof[i], dof[j]] += Ke[i, j]

        self.K = K_total

    def add_sets(self, set_name: str, set_data: list[int]):
        """
        添加集合

        Args:
            set_name: 集合名称
            set_data: 集合数据

        """
        self.sets[set_name] = set_data

    def apply_boundary_condition(self, set_name, direct: str):
        node_ids = self.sets[set_name]
        if direct == 'X':
            fixed_dof = [2 * i for i in node_ids]
            self.fixed_dof += fixed_dof
        elif direct == 'Y':
            fixed_dof = [2 * i + 1 for i in node_ids]
            self.fixed_dof += fixed_dof
        elif direct == 'ALL':
            fixed_dof = [2 * i for i in node_ids] + [2 * i + 1 for i in node_ids]
            self.fixed_dof += fixed_dof

    def add_loads(self, set_name, f_x: float = 0.0, f_y: float = 0.0):
        """在边界上施加节点力"""
        node_ids = self.sets[set_name]
        if len(node_ids) == 1:
            # 如果只有一个节点，那么施加的是集中力
            x_dof_id = 2 * node_ids[0]
            y_dof_id = 2 * node_ids[0] + 1
            self.F[x_dof_id] += f_x
            self.F[y_dof_id] += f_y
        else:
            # 如果有多个节点，则施加的是均布力
            x_dof_ids = [2 * i for i in node_ids]
            y_dof_ids = [2 * i + 1 for i in node_ids]
            self.F[x_dof_ids[1:-1]] += f_x
            self.F[y_dof_ids[1:-1]] += f_y
            self.F[x_dof_ids[0]] += 0.5 * f_x
            self.F[x_dof_ids[-1]] += 0.5 * f_x
            self.F[y_dof_ids[0]] += 0.5 * f_y
            self.F[y_dof_ids[-1]] += 0.5 * f_y

    def solve(self):
        self.cal_K_total()

        n = self.num_dof
        free_dof = list((set(range(n)).difference(self.fixed_dof)))

        # 删除已经固定的自由度
        K_ff = self.K[np.ix_(free_dof, free_dof)]
        F_f = np.array([self.F[i] for i in free_dof])

        # K_ff = csr_matrix(K_ff)
        U_f = np.linalg.solve(K_ff, F_f)
        U_f[np.abs(U_f) < 1e-8] = 0

        self.U[free_dof] = U_f
        self.U = self.U.reshape(-1, 2)

    def cal_stress_field(self):
        stress = calculate_element_stress(U=self.U.flatten(),
                                          Element=self.Element,
                                          Node=self.Node,
                                          node_dof_indices=self.node_dof_indices,
                                          D=self.D)
        self.stress = stress

    def display_mesh(self):
        visualize_2d_mesh_plus(self.Node,
                               self.Element,
                               self.node_dof_indices)

    def show_deformation(self):
        Node_new = self.Node + self.U
        visualize_2d_mesh_plus(Node_new,
                               self.Element,
                               self.node_dof_indices)

    def show_contour(self, component: int = 0):
        Node_new = self.Node + self.U
        self.cal_stress_field()
        node_stresses = interpolate_stress_to_nodes(Element=self.Element,
                                                    element_stresses=self.stress,
                                                    num_nodes=self.num_nodes)
        plot_stress_cloud(self.Node, self.Element, node_stresses, stress_component=component, coord_new=Node_new)
