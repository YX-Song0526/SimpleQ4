import numpy as np
from abc import ABC, abstractmethod
from visualize import visualize_2d_mesh_plus


class Mesh(ABC):
    def __init__(self):
        self.Node = None
        self.Element = None
        self.num_nodes = None
        self.num_elements = None
        self.node_dof_indices = None
        self.sets = {}
        self.params = {}

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def fill(self):
        pass


class Rectangle(Mesh):
    def __init__(self,
                 x1: float,
                 x2: float,
                 y1: float,
                 y2: float,
                 Nx: int,
                 Ny: int):
        super().__init__()
        self.params['x1'] = x1
        self.params['x2'] = x2
        self.params['y1'] = y1
        self.params['y2'] = y2
        self.params['Nx'] = Nx
        self.params['Ny'] = Ny
        self.update(x1=x1, x2=x2, y1=y1, y2=y2, Nx=Nx, Ny=Ny)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
        self.fill()

    def fill(self):
        x1 = self.params['x1']
        x2 = self.params['x2']
        y1 = self.params['y1']
        y2 = self.params['y2']
        Nx = self.params['Nx']
        Ny = self.params['Ny']

        self.num_nodes = Nx * Ny
        self.num_elements = (Nx - 1) * (Ny - 1)
        self.node_dof_indices = np.arange(0, 2 * self.num_nodes).reshape(-1, 2)

        dx = (x2 - x1) / (Nx - 1)
        dy = (y2 - y1) / (Ny - 1)

        # 填充节点坐标矩阵
        self.Node = np.array([[x1 + i * dx, y1 + j * dy] for j in range(Ny) for i in range(Nx)])
        self.sets['lower'] = np.arange(0, Nx).tolist()
        self.sets['upper'] = np.arange(Nx * (Ny - 1), Nx * Ny).tolist()
        self.sets['left'] = np.arange(0, Nx * Ny, Nx).tolist()
        self.sets['right'] = np.arange(Nx - 1, Nx * Ny, Nx).tolist()

        elements = np.zeros((self.num_elements, 4), dtype=int)

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

        self.Element = elements

    def display(self):
        visualize_2d_mesh_plus(self.Node, self.Element, self.node_dof_indices)


class Circle(Mesh):
    def __init__(self,
                 R: float,
                 N1: int,
                 N2: int):
        super().__init__()
        self.params['N1'] = N1
        self.params['N2'] = N2

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value
        self.fill()

    
# rect = Rectangle(0, 2, 1, 3, 9, 9)
# print(rect.sets['left'])
# print(rect.sets['right'])
# print(rect.sets['upper'])
# rect.display()
