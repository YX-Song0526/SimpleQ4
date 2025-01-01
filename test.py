import numpy as np

from plane_problems import PlaneElastic
from mesh import Rectangle
from visualize import visualize_2d_mesh_plus
from parameters import *

mesh = Rectangle(x_start=2,
                 x_end=3,
                 y_start=0,
                 y_end=4,
                 Nx=10,
                 Ny=20)

Node = mesh.coord
Element = mesh.elements
ndf = mesh.node_dof_idx

model = PlaneElastic(problem_type=PLANE_STRESS)

model.add_nodes(Node)
model.add_elements(Element)

model.display_mesh()

# print(np.arange(0, 200, 10))

# 设置材料参数
model.set_material(E=69e9, niu=0.28)

# 添加边界集合
inside = list(np.arange(0, 200, 10))
bottom = list(np.arange(0, 10))

model.add_sets('inside', inside)
model.add_sets('bottom', bottom)

# print(model1.sets['left'])

model.apply_boundary_condition('bottom', direct='ALL')
# print(model1.fixed_dof)

model.add_loads('inside', f_x=100000000)

model.solve()

model.show_contour(component=1)
