import numpy as np


class Rectangle:
    def __init__(self,
                 x_start: float,
                 x_end: float,
                 y_start: float,
                 y_end: float,
                 Nx: int,
                 Ny: int):
        self.num_nodes = Nx * Ny
        self.num_cells = (Nx - 1) * (Ny - 1)
