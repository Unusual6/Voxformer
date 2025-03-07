# xming
# set DISPLAY=localhost:0.0
# ssh -Y -v -P 48300 root@10.134.151.254

# 在服务器上安装Xvfb（虚拟X服务器），它能提供一个虚拟的图形环境，
# 使Mayavi可以进行离屏渲染并保存图片，而不需要真正的显示器
# xvfb-run -a python tools/vis.py


# from operator import gt
import pickle
import numpy as np
from omegaconf import DictConfig
import hydra
from mayavi import mlab



def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid


def draw(
    voxels,
    T_velo_2_cam,
    vox_origin,
    fov_mask,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords[fov_mask, :]

    # Get the voxels outside FOV
    outfov_grid_coords = grid_coords[~fov_mask, :]

    # Remove empty and unknown voxels
    fov_voxels = fov_grid_coords[
        (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 255)
    ]
    outfov_voxels = outfov_grid_coords[
        (outfov_grid_coords[:, 3] > 0) & (outfov_grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

    # Draw the camera
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    # Draw occupied outside FOV voxels
    plt_plot_outfov = mlab.points3d(
        outfov_voxels[:, 0],
        outfov_voxels[:, 1],
        outfov_voxels[:, 2],
        outfov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    colors = np.array(
        [
            [100, 150, 245, 255],
            [100, 230, 245, 255],
            [30, 60, 150, 255],
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255],
        ]
    ).astype(np.uint8)

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_outfov.glyph.scale_mode = "scale_by_vector"

    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    outfov_colors = colors
    outfov_colors[:, :3] = outfov_colors[:, :3] // 3 * 2
    plt_plot_outfov.module_manager.scalar_lut_manager.lut.table = outfov_colors

    mlab.show()

def draw1(
    voxels,
    T_velo_2_cam,
    vox_origin,
    img_size,
    f,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
):


    # Compute the coordinates of the mesh representing camera
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # 只保留非空体素
    valid_voxels = grid_coords[
        (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))

    # Draw the camera
    mlab.triangular_mesh(
        x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5
    )

    # Draw all occupied voxels（不再区分 FOV 内外）
    plt_plot = mlab.points3d(
        valid_voxels[:, 0],
        valid_voxels[:, 1],
        valid_voxels[:, 2],
        valid_voxels[:, 3],  # 颜色按类别编码
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    colors = np.array(
        [
            [100, 150, 245, 255],
            [100, 230, 245, 255],
            [30, 60, 150, 255],
            [80, 30, 180, 255],
            [100, 80, 250, 255],
            [255, 30, 30, 255],
            [255, 40, 200, 255],
            [150, 30, 90, 255],
            [255, 0, 255, 255],
            [255, 150, 255, 255],
            [75, 0, 75, 255],
            [175, 0, 75, 255],
            [255, 200, 0, 255],
            [255, 120, 50, 255],
            [0, 175, 0, 255],
            [135, 60, 0, 255],
            [150, 240, 80, 255],
            [255, 240, 150, 255],
            [255, 0, 0, 255],
        ]
    ).astype(np.uint8)

    plt_plot.glyph.scale_mode = "scale_by_vector"
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    mlab.savefig('ssc1.png')  # 保存到 /tmp
    print("Image saved to /tmp/ssc.png")

    # mlab.show()

import numpy as np
import open3d as o3d

def save_voxel_grid(
    voxels,
    T_velo_2_cam,
    vox_origin,
    voxel_size=0.2,
    output_file="voxel_grid.ply"
):
    """
    保存体素网格为 PLY 文件格式
    """
    # 计算体素的网格坐标
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # 只保留非空体素
    valid_voxels = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    valid_voxels = valid_voxels[(valid_voxels[:, 3] > 0) & (valid_voxels[:, 3] < 255)]

    # 体素颜色映射 (RGB)
    colors = np.array([
        [100, 150, 245],
        [100, 230, 245],
        [30, 60, 150],
        [80, 30, 180],
        [100, 80, 250],
        [255, 30, 30],
        [255, 40, 200],
        [150, 30, 90],
        [255, 0, 255],
        [255, 150, 255],
        [75, 0, 75],
        [175, 0, 75],
        [255, 200, 0],
        [255, 120, 50],
        [0, 175, 0],
        [135, 60, 0],
        [150, 240, 80],
        [255, 240, 150],
        [255, 0, 0],
    ]).astype(np.uint8)

    # 体素类别映射到颜色
    voxel_colors = colors[(valid_voxels[:, 3] - 1).astype(int) % len(colors)] / 255.0

    # 使用 open3d 创建一个体素网格
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.origin = np.array([vox_origin[0], vox_origin[1], vox_origin[2]])
    voxel_grid.voxel_size = voxel_size

    # 将体素数据转换为 open3d 的体素
    for voxel in valid_voxels:
        voxel_center = voxel[:3]
        voxel_color = voxel_colors[int(voxel[3])]

        # 添加体素到 voxel_grid
        voxel_grid.add_voxel(voxel_center, voxel_color)

    # 保存为 PLY 文件
    o3d.io.write_voxel_grid(output_file, voxel_grid)
    print(f"Voxel grid saved to {output_file}")


import numpy as np
import open3d as o3d
import time

def save_voxel_grid_as_image(
    voxels,
    T_velo_2_cam,
    vox_origin,
    voxel_size=0.2,
    output_image="voxel_grid_image.png",
    width=800,  # 图片宽度
    height=800  # 图片高度
):
    """
    保存体素网格的2D截图
    """
    # 计算体素网格坐标
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )
    # 将体素数据与预测类别合并，形状为 (N, 4)
    valid_voxels = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    valid_voxels = valid_voxels[(valid_voxels[:, 3] > 0) & (valid_voxels[:, 3] < 255)]
    
    if valid_voxels.shape[0] == 0:
        print("没有有效体素数据！")
        return
    
    # 定义颜色映射 (RGB)
    colors = np.array([
        [100, 150, 245],
        [100, 230, 245],
        [30, 60, 150],
        [80, 30, 180],
        [100, 80, 250],
        [255, 30, 30],
        [255, 40, 200],
        [150, 30, 90],
        [255, 0, 255],
        [255, 150, 255],
        [75, 0, 75],
        [175, 0, 75],
        [255, 200, 0],
        [255, 120, 50],
        [0, 175, 0],
        [135, 60, 0],
        [150, 240, 80],
        [255, 240, 150],
        [255, 0, 0],
    ]).astype(np.uint8)
    # 根据体素类别映射颜色（注意类别从1开始）
    voxel_colors = colors[(valid_voxels[:, 3] - 1).astype(int) % len(colors)] / 255.0

    # 创建点云对象，作为体素数据的展示
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_voxels[:, :3])
    point_cloud.colors = o3d.utility.Vector3dVector(voxel_colors)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)
    vis.add_geometry(point_cloud)
    
    # 设置背景颜色为白色
    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([1, 1, 1])
    
    # 调整相机视角
    ctr = vis.get_view_control()
    # 以点云包围盒中心作为观察中心
    bbox = point_cloud.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center())
    # 设置前视角和上方向（可根据实际数据调整）
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.5)  # 缩放比例，根据体素范围适当调整
    
    # 让渲染器充分更新（等待1秒）
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1)  # 等待足够时间确保窗口渲染完成
    
    # 截图保存
    vis.capture_screen_image(output_image)
    print(f"体素网格截图已保存为 {output_image}")
    
    vis.destroy_window()


# @hydra.main(config_path=None)
def main():
    scan = '/root/VoxFormer/test.pkl'
    with open(scan, "rb") as handle:
        b = pickle.load(handle)

    # fov_mask_1 = b["fov_mask_1"]
    T_velo_2_cam = b[0]["T_velo_2_cam"]
    vox_origin = np.array([0, -25.6, -2])

    y_pred = b[0]["y_pred"][0]
    # y_pred = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
    # y_pred = np.argmax(y_pred, axis=1)

    # save_voxel_grid(y_pred, T_velo_2_cam, vox_origin)
    # save_voxel_grid_as_image(y_pred, T_velo_2_cam, vox_origin)

    draw1(
        y_pred,
        T_velo_2_cam,
        vox_origin,
        # fov_mask_1,
        img_size=(1220, 370),
        f=707.0912,
        voxel_size=0.2,
        d=7,
    )


if __name__ == "__main__":
    main()
