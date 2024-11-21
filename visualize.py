import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
import matplotlib.tri as tri
import numpy as np
from matplotlib.text import Annotation


def visualize_2d_mesh(coord, elements):
    """
    绘制二维有限元网格。
    Args:
        coord: 节点坐标矩阵 (num_nodes, 2)
        elements: 单元连接矩阵 (num_elements, 4)
    """
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制单元
    for element in elements:
        vertices = coord[element, :]  # 获取该单元的节点坐标
        polygon = Polygon(vertices, edgecolor='k', facecolor='lightblue', linewidth=1, alpha=0.6)
        ax.add_patch(polygon)

    # 设置图形属性
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Finite Element Mesh')
    ax.set_xlim([coord[:, 0].min() - 0.5, coord[:, 0].max() + 0.5])
    ax.set_ylim([coord[:, 1].min() - 0.1, coord[:, 1].max() + 1])

    # plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()


def plot_stress_cloud(coord, elements, node_stresses, stress_component=0, coord_new=None):
    """
    绘制应力云图，支持变形后的坐标。
    Args:
        coord: 初始坐标 (num_nodes, 2)
        elements: 单元连接矩阵 (num_elements, 4)
        node_stresses: 节点应力矩阵 (num_nodes, 3)
        stress_component: 要绘制的应力分量索引 (0: σx, 1: σy, 2: τxy)
        coord_new: 变形后的坐标 (num_nodes, 2), 如果为 None，则使用初始坐标
    """
    # 提取要绘制的应力分量
    stress_values = node_stresses[:, stress_component]

    # 使用变形后的坐标
    if coord_new is None:
        coord_to_plot = coord
    else:
        coord_to_plot = coord_new

    # 将四边形单元拆分为三角形单元
    triangles = []
    for quad in elements:
        triangles.append([quad[0], quad[1], quad[2]])
        triangles.append([quad[0], quad[2], quad[3]])

    triangles = np.array(triangles)  # 转为 NumPy 数组

    # 创建三角剖分
    triangulation = tri.Triangulation(coord_to_plot[:, 0], coord_to_plot[:, 1], triangles)

    # 绘制云图
    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(triangulation, stress_values, cmap='jet', alpha=0.8)
    plt.colorbar(contour, label="Stress")
    plt.title(f"Stress Cloud (Component {stress_component})")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.gca().set_aspect('equal')

    # 可选：叠加网格线
    for quad in elements:
        vertices = coord_to_plot[quad]
        plt.plot(*vertices[[0, 1, 2, 3, 0]].T, 'k-', linewidth=0.5)  # 画四边形网格线

    plt.show()


def plot_comparison(coord, coord_new, elements, overturn=False):
    """
    绘制变形前后的网格，用不同颜色区分。

    Args:
        coord: 初始节点坐标 (num_nodes, 2)
        coord_new: 变形后节点坐标 (num_nodes, 2)
        elements: 单元连接矩阵 (num_elements, 4)
        overturn: 是否交换 X 和 Y 坐标
    """
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 如果需要交换坐标轴
    if overturn:
        coord = coord[:, [1, 0]]
        coord_new = coord_new[:, [1, 0]]

    # 绘制变形前的网格（淡蓝色）
    for element in elements:
        vertices = coord[element, :]
        polygon = Polygon(vertices, edgecolor='k', facecolor='lightblue', linewidth=1, alpha=0.6)
        ax.add_patch(polygon)

    # 绘制变形后的网格（红色）
    for element in elements:
        vertices = coord_new[element, :]
        polygon = Polygon(vertices, edgecolor='k', facecolor='lightcoral', linewidth=1, alpha=0.6)
        ax.add_patch(polygon)

    # 手动添加图例
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='k', linewidth=1, label='Original Mesh'),
        Patch(facecolor='lightcoral', edgecolor='k', linewidth=1, label='Deformed Mesh')
    ]
    ax.legend(handles=legend_elements, loc='best')

    # 设置图形属性
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Comparison of Original and Deformed Mesh')
    ax.set_xlim([min(coord[:, 0].min(), coord_new[:, 0].min()) - 0.5,
                 max(coord[:, 0].max(), coord_new[:, 0].max()) + 0.5])
    ax.set_ylim([min(coord[:, 1].min(), coord_new[:, 1].min()) - 0.1,
                 max(coord[:, 1].max(), coord_new[:, 1].max()) + 1])

    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.show()


def visualize_2d_mesh_plus(coord, elements, node_dof_idx):
    """
    绘制二维有限元网格，支持鼠标悬停显示节点信息。

    Args:
        coord: 节点坐标矩阵 (num_nodes, 2)
        elements: 单元连接矩阵 (num_elements, 4)
        node_dof_idx: 节点自由度索引矩阵 (num_nodes, 2)
    """
    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(10, 5))

    # 绘制单元
    for element in elements:
        vertices = coord[element, :]  # 获取该单元的节点坐标
        polygon = Polygon(vertices, edgecolor='k', facecolor='lightblue', linewidth=1, alpha=0.6)
        ax.add_patch(polygon)

    # 绘制节点
    scat = ax.scatter(coord[:, 0], coord[:, 1], color='k', s=0.2, zorder=5)

    # 设置图形属性
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Finite Element Mesh')
    ax.set_xlim([coord[:, 0].min() - 0.5, coord[:, 0].max() + 0.5])
    ax.set_ylim([coord[:, 1].min() - 0.1, coord[:, 1].max() + 1])

    # 创建注释对象
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(10, 10),
        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    annot.set_visible(False)

    def update_annot(ind):
        """更新注释内容和位置"""
        idx = ind["ind"][0]  # 获取鼠标悬停的点索引
        pos = coord[idx]
        dof = node_dof_idx[idx]
        annot.xy = pos
        text = f"Node: {idx}\nCoord: ({pos[0]:.2f}, {pos[1]:.2f})\nDOF: ({dof[0]}, {dof[1]})"
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.8)

    def on_hover(event):
        """鼠标悬停事件处理"""
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scat.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    # 连接鼠标事件
    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    plt.show()
