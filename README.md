# [Python] SimpleQ4

## 介绍

该代码为基于**四节点四边形等参元**的有限元计算程序，主要实现了：

- 矩形网格划分及节点坐标矩阵和单元连接矩阵的导出
- 节点位移的计算
- 模态分析
- 变形结果和应力分布的可视化

## 使用样例

### 1. 一端固支的悬臂梁

以该项目下的`example.py`为例：

```python
from stiffness_matrix import * 
# 从preprocess.py中导入节点坐标矩阵等信息，也可以采用其他方法如使用专业网格划分软件等生成网格并处理得到这些信息
from preprocess import coord, elements, num_dof, num_nodes, interpolate_stress_to_nodes  
from visualize import *

# 调用函数得到总体刚度矩阵
K = cal_K_total(coord, elements)

# 初始化力向量和位移向量
F = np.zeros(num_dof)
U = np.zeros(num_dof)

# 定义载荷
F[-2] = -0.02
F[142] = 0.07

# 删除被固定的自由度
K_reduced = K[12:, 12:]

# 求解位移
F_reduced = F[12:]
U_reduced = np.linalg.solve(K_reduced, F_reduced)

# 后处理
U[12:] = U_reduced
U1 = U.copy()
U = U.reshape(num_nodes, 2)
coord_new = coord + U
coord_1 = coord_new.copy()

# 调转方向
temp = coord_new.copy()
coord_new[:, 0] = temp[:, 1]
coord_new[:, 1] = temp[:, 0]

# 输出位移矩阵
print(U)

# 可视化
plot_comparison(coord=coord, coord_new=coord_1, elements=elements, overturn=True)

# 计算应力场
element_stresses = calculate_element_stress(U.flatten(), elements, coord)
node_stresses = interpolate_stress_to_nodes(elements, element_stresses, num_nodes)

# 绘制变形后的应力云图
plot_stress_cloud(coord, elements, node_stresses, stress_component=1, coord_new=coord_new)
```

节点位移：

```cmd
[[ 0.00000000e+00  0.00000000e+00]
 [ 0.00000000e+00  0.00000000e+00]
 [ 0.00000000e+00  0.00000000e+00]
 ......
 [-2.41848664e+00  8.70643155e-02]
 [-2.42334548e+00  2.68954831e-01]
 [-2.43056528e+00  4.56239250e-01]]
```

可视化结果：

![image](https://github.com/user-attachments/assets/0854f6eb-9f15-414c-a2f8-a80e8ae321f4)
![image](https://github.com/user-attachments/assets/c6be406c-22a9-4be5-a489-52e7132e90d7)


### 2. 两端固支悬臂梁的模态分析

以该项目下的`example2.py`为例

```python
import numpy as np
from scipy.linalg import eigh
from stiffness_matrix import cal_K_total, cal_M_total, calculate_element_stress
from preprocess import coord, elements, num_nodes, num_dof, interpolate_stress_to_nodes
from visualize import plot_stress_cloud

# 计算全局刚度矩阵和全局质量矩阵
K = cal_K_total(coord, elements)
M = cal_M_total(coord, elements)

# 两端固定边界条件
K_reduced = K[12:-12, 12:-12]
M_reduced = M[12:-12, 12:-12]

# 求解广义特征值问题
eigenvalues, eigenvectors = eigh(K_reduced, M_reduced)

# 计算模态频率（以 Hz 表示）
frequencies = np.sqrt(eigenvalues) / (2 * np.pi)

U = np.zeros(num_dof)
U[12:-12] = eigenvectors[:, 2]  # 第3阶振型
U = U.reshape(num_nodes, 2)

coord_new = coord + 10 * U  # 将位移放大十倍
print(U)

# 计算应力场
element_stresses = calculate_element_stress(U.flatten(), elements, coord)
node_stresses = interpolate_stress_to_nodes(elements, element_stresses, num_nodes)

# 绘制变形后的应力云图
plot_stress_cloud(coord, elements, node_stresses, stress_component=1, coord_new=coord_new)

```

可视化结果：

![image](https://github.com/user-attachments/assets/67cfd83c-76fe-40b9-9141-665048d554c5)


