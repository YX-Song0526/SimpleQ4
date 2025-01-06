from plane_problems import PlaneElastic
from parameters import *


model1 = PlaneElastic(problem_type=PLANE_STRESS)

# 导入网格
input_file = 'a_b_1.inp'
model1.load_mesh(input_file)

# 设置材料参数
model1.set_material(E=69e9, niu=0.28)

# 添加边界集合
left_boundary = [2, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3]
right_boundary = [9, 93, 94, 95, 96, 97, 98, 99, 100, 101, 10]
model1.add_sets('left', left_boundary)
model1.add_sets('right', right_boundary)
# print(model1.sets['left'])

model1.apply_boundary_condition('left', direct='ALL')
# print(model1.fixed_dof)

model1.add_loads('right', f_x=100000000000)

model1.solve()

model1.show_contour(component=1)
