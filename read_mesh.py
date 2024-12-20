import re
import numpy as np


def read_inp(file_path, dimension=2):
    """
    从 .inp 文件中提取节点和元素信息，并根据维度选择处理二维或三维网格。

    :param file_path: 输入文件路径
    :param dimension: 网格的维度，2 或 3，默认为 2
    :return: nodes_matrix, elements_matrix
    """
    nodes = []
    elements = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 标志位，用于区分 *Node 和 *Element 区域
    reading_nodes = False
    reading_elements = False

    for line in lines:
        line = line.strip()

        if line.startswith("*Node"):
            reading_nodes = True
            reading_elements = False
            continue

        if line.startswith("*Element"):
            reading_elements = True
            reading_nodes = False
            continue

        if line.startswith("*") and reading_elements:
            # 遇到其他部分时，停止读取元素数据
            reading_elements = False

        if reading_nodes:
            # 根据维度选择不同的节点提取方式
            if dimension == 2:
                # 读取二维网格的节点，格式为 ID, x, y
                node_data = re.findall(r'\d+,\s*([\d.-]+),\s*([\d.-]+)', line)
                # print(node_data)
                if node_data:
                    for node in node_data:
                        x, y = float(node[0]), float(node[1])
                        nodes.append([x, y])  # 只存储二维坐标
            elif dimension == 3:
                # 读取三维网格的节点，格式为 ID, x, y, z
                node_data = re.findall(r'\d+,\s*([\d.-]+),\s*([\d.-]+),\s*([\d.-]+)', line)
                if node_data:
                    for node in node_data:
                        x, y, z = float(node[0]), float(node[1]), float(node[2])
                        nodes.append([x, y, z])  # 存储三维坐标

        if reading_elements:
            # 提取元素的节点连接信息
            # 匹配一个元素行：ID, 节点ID1, 节点ID2, 节点ID3, ..., 节点IDn
            element_data = re.findall(r'\d+\s*,\s*((?:\d+\s*,\s*)*\d+)', line)
            if element_data:
                for element in element_data:
                    # 将节点ID从字符串转换为整数，并减去1（假设从1开始的ID转为从0开始的索引）
                    node_ids = [int(node_id) - 1 for node_id in element.split(',')]
                    elements.append(node_ids)

    # 将提取的节点和元素数据转换为 NumPy 数组（矩阵）
    nodes_matrix = np.array(nodes)
    elements_matrix = np.array(elements)

    return nodes_matrix, elements_matrix

