from plane_problems import PlaneElastic
from parameters import *
from test2 import nodes, elements

model1 = PlaneElastic(problem_type=PLANE_STRESS)

# 导入网格
model1.add_nodes(nodes)
model1.add_elements(elements)

# 设置材料参数
model1.set_material(E=210e9, niu=0.28)

# 添加边界集合
top = [260]
bottom = [276]
middle = [268, 284]
model1.add_sets('top', top)
model1.add_sets('bottom', bottom)
model1.add_sets('middle', middle)
# print(model1.sets['left'])

model1.apply_boundary_condition('middle', direct='Y')
# print(model1.fixed_dof)

model1.add_loads('top', f_y=10000000000)
model1.add_loads('bottom', f_y=-10000000000)


model1.solve()

model1.show_contour(component=0)
model1.show_contour(component=1)
model1.show_contour(component=2)