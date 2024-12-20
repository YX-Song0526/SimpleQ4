import numpy as np
from visualize import visualize_2d_mesh


def fill_coord_mat(x_start,
                   x_end,
                   y_start,
                   y_end,
                   Nx,
                   Ny):
    """填充节点坐标矩阵"""

    dx = (x_end - x_start) / (Nx - 1)
    dy = (y_end - y_start) / (Ny - 1)

    coord = np.array([[x_start + i * dx, y_start + j * dy]
                      for j in range(Ny) for i in range(Nx)])

    return coord


def fill_elements_mat(Nx, Ny):
    """
    填充单元节点索引矩阵
    """
    num_elements = (Nx - 1) * (Ny - 1)
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

    return elements


class Rectangle:
    def __init__(self,
                 x_start: float,
                 x_end: float,
                 y_start: float,
                 y_end: float,
                 Nx: int,
                 Ny: int):
        dx = (x_end - x_start) / (Nx - 1)
        dy = (y_end - y_start) / (Ny - 1)

        self.num_nodes = Nx * Ny
        self.num_elements = (Nx - 1) * (Ny - 1)
        self.coord = fill_coord_mat(x_start, x_end, y_start, y_end, Nx, Ny)
        self.elements = fill_elements_mat(Nx, Ny)
        self.node_dof_idx = np.arange(0, 2 * self.num_nodes).reshape(-1, 2)


mesh = Rectangle(0.0,
                 5.0,
                 0.0,
                 10.0,
                 5,
                 10)

coord = mesh.coord
elements = mesh.elements

visualize_2d_mesh(coord, elements)
