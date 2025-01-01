import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def ids_1dto2d(ids, M=10, N=10):
    i = ids // N
    j = ids - N * i
    return i, j

def render_adv(file):
    agents = file['agents']
    orders = file['orders']
    actions = file['actions']
    adv_agents_id = file['adv_agents']
    adv_agents = agents[:, adv_agents_id]

    num_frames = agents.shape[0]

    fig, ax = plt.subplots(figsize=(10, 10),  dpi=100) #250
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.set_aspect('equal', adjustable='box')

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)

    # 创建蓝色方形点的散点图
    scatter_blue = ax.scatter([], [], color='blue', s=1000, marker='s')  # 蓝色方形点
    scatter_green = ax.scatter([], [], color='green', s=1000, marker='s')  # 绿色方形点
    # 创建红色圆点的散点图
    scatter_red = ax.scatter([], [], color='red', s=500, marker='o')  # 红色圆点
    annotations = []  # 用来存储标记数字的文本对象
    arrow_patches = []

    def init():
        # 初始化两个散点图（蓝色方形和红色圆点）
        scatter_blue.set_offsets(np.empty((0, 2)))
        scatter_red.set_offsets(np.empty((0, 2)))
        scatter_green.set_offsets(np.empty((0, 2)))
        # 清除文本标记
        for annotation in annotations:
            annotation.remove()
        annotations.clear()  # 清空文本对象列表
        arrow_patches.clear()
        return scatter_blue, scatter_red, scatter_green

    def update(frame):
         # 移除上一次帧的箭头
        for patch in arrow_patches:
            patch.remove()
        arrow_patches.clear()

        current_blue_points = agents[frame]
        scatter_blue.set_offsets(current_blue_points)

        current_green_points = adv_agents[frame]
        scatter_green.set_offsets(current_green_points)

        for annotation in annotations:
            annotation.remove()
        annotations.clear()

        new_points = []
        for i in range(len(orders[frame])):
            if orders[frame][i] > 0:
                x,y=ids_1dto2d(i)
                new_points.append((x,y))
                annotation = ax.text(x, y, str(int(orders[frame][i])), color='white', fontsize=12, ha='center', va='center')
                annotations.append(annotation)

        if len(new_points) == 0:
            new_points = np.empty((0, 2))

        scatter_red.set_offsets(new_points)

        # 绘制箭头
        direction_map = {0: (0, 0.5), 1: (0, -0.5), 2: (-0.5, 0), 3: (0.5, 0)}
        for i, (x, y) in enumerate(current_blue_points):
            act = actions[frame, i]
            if act in direction_map:
                dx, dy = direction_map[act]
                if i in adv_agents_id:
                    color='green'
                else:
                    color='blue'
                arrow = ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, color=color)
                arrow_patches.append(arrow)

        ax.set_title(f"Frame {frame + 1} of {num_frames}")
        return scatter_blue, scatter_red, scatter_green, *annotations

    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=500)
    ani.save("rollout_adv.gif", writer='imagemagick')
    plt.show()

if __name__ == '__main__':
    file = np.load('render/rollout_adv_100.npz')
    render_adv(file)