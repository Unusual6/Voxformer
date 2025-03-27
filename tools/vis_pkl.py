import pickle
import numpy as np
from omegaconf import DictConfig
import hydra
from mayavi import mlab
import argparse
import os
import open3d as o3d

# 设置 offscreen 渲染
mlab.options.offscreen = True

def get_grid_coords(dims, resolution):
    # 原函数不变
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    sensor_pose = 10
    g_zz = np.arange(0, dims[2] + 1)
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)
    coords_grid = (coords_grid * resolution) + resolution / 2
    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)
    return coords_grid

def draw1(id, pred, voxels, T_velo_2_cam, vox_origin, img_size, f, voxel_size=0.2, d=7):
    # 原函数不变
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array([[0, 0, 0], [x, y, d], [-x, y, d], [-x, -y, d], [x, -y, d]])
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [(0, 1, 2), (0, 1, 4), (0, 3, 4), (0, 2, 3)]

    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    valid_voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]

    figure = mlab.figure(size=(1400, 1400), bgcolor=(1, 1, 1))
    mlab.triangular_mesh(x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5)
    plt_plot = mlab.points3d(
        valid_voxels[:, 0],
        valid_voxels[:, 1],
        valid_voxels[:, 2],
        valid_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )
    colors = np.array([
        [100, 150, 245, 255], [100, 230, 245, 255], [30, 60, 150, 255],
        [80, 30, 180, 255], [100, 80, 250, 255], [255, 30, 30, 255],
        [255, 40, 200, 255], [150, 30, 90, 255], [255, 0, 255, 255],
        [255, 150, 255, 255], [75, 0, 75, 255], [175, 0, 75, 255],
        [255, 200, 0, 255], [255, 120, 50, 255], [0, 175, 0, 255],
        [135, 60, 0, 255], [150, 240, 80, 255], [255, 240, 150, 255],
        [255, 0, 0, 255],
    ]).astype(np.uint8)
    plt_plot.glyph.scale_mode = "scale_by_vector"
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors
    mlab.savefig(f'vis_output/{id}_{pred}_ssc.png')
    print(pred,"Image saved to vis_output")

def save_voxel_grid(
    voxels,
    vox_origin,
    voxel_size=0.2,
    output_file="voxel_grid1.ply"
):
    """
    Save voxel grid as PLY file format.
    """
    # 计算体素网格的坐标
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # 只保留有效体素
    valid_voxels = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    valid_voxels = valid_voxels[(valid_voxels[:, 3] > 0) & (valid_voxels[:, 3] < 255)]

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
    point_cloud.points = o3d.utility.Vector3dVector(valid_voxels[:, :3])  # 点的位置
    point_cloud.colors = o3d.utility.Vector3dVector(voxel_colors)  # 点的颜色

    # 保存为 .ply 文件
    o3d.io.write_point_cloud(output_file, point_cloud)

    print(f"Voxel grid saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qt-plugin', default=None, help='Qt platform plugin')
    args = parser.parse_args()
    if args.qt_plugin:
        os.environ["QT_QPA_PLATFORM"] = args.qt_plugin

    scan = '/root/VoxFormer/test_output.pkl'
    with open(scan, "rb") as handle:
        b = pickle.load(handle)
    
    print("len B:",len(b))
    for i in b:
        T_velo_2_cam = i["T_velo_2_cam"]
        vox_origin = np.array([0, -25.6, -2])
        y_true = i["y_true"][0]
        y_pred = i["y_pred"][0]
        save_voxel_grid(y_pred, vox_origin)
        id = i['id']
        draw1(id, "pred", y_pred, T_velo_2_cam, vox_origin, img_size=(1220, 370), f=707.0912, voxel_size=0.2, d=7)
        draw1(id, "true", y_true, T_velo_2_cam, vox_origin, img_size=(1220, 370), f=707.0912, voxel_size=0.2, d=7)

if __name__ == "__main__":
    main()