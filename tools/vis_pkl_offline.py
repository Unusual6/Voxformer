import pickle
import numpy as np
from mayavi import mlab
import argparse
import os
from multiprocessing import Pool, cpu_count
import functools

# 设置 offscreen 渲染
mlab.options.offscreen = True

def get_grid_coords(dims, resolution):
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    coords_grid = (coords_grid * resolution) + resolution / 2
    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    return temp

def draw1(id, pred, voxels, T_velo_2_cam, vox_origin, img_size, f, voxel_size=0.2, d=7):
    # 计算三角形网格点
    x = d * img_size[0] / (2 * f)
    y = d * img_size[1] / (2 * f)
    tri_points = np.array([[0, 0, 0], [x, y, d], [-x, y, d], [-x, -y, d], [x, -y, d]])
    tri_points = np.hstack([tri_points, np.ones((5, 1))])
    tri_points = (np.linalg.inv(T_velo_2_cam) @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [(0, 1, 2), (0, 1, 4), (0, 3, 4), (0, 2, 3)]

    # 计算网格坐标
    grid_coords = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    valid_voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]

    # 创建独立的 figure，每个进程独立渲染
    figure = mlab.figure(size=(1000, 1000), bgcolor=(1, 1, 1))

    # 绘制三角形网格
    mlab.triangular_mesh(x, y, z, triangles, representation="wireframe", color=(0, 0, 0), line_width=5, figure=figure)

    # 绘制体素点
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
        figure=figure,
    )
    colors = np.array([
        [100, 150, 245, 255], [100, 230, 245, 255], [30, 60, 150, 255],
        [80, 30, 180, 255], [100, 80, 250, 255], [255, 30, 30, 255],
        [255, 40, 200, 255], [150, 30, 90, 255], [255, 0, 255, 255],
        [255, 150, 255, 255], [75, 0, 75, 255], [175, 0, 75, 255],
        [255, 200, 0, 255], [255, 120, 50, 255], [0, 175, 0, 255],
        [135, 60, 0, 255], [150, 240, 80, 255], [255, 240, 150, 255],
        [255, 0, 0, 255],
    ], dtype=np.uint8)
    plt_plot.glyph.scale_mode = "scale_by_vector"
    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    # 保存图片
    output_path = f'/root/VoxFormer/S_16epoch/{id}_{pred}_ssc.png'
    mlab.savefig(output_path, figure=figure)
    print(f"{pred} Image saved to {output_path}")

    # 清理当前进程的 figure
    mlab.close(figure)

def process_item(item, vox_origin, img_size, f, voxel_size=0.2, d=7):
    """处理单个 item 的 pred 和 true"""
    T_velo_2_cam = item["T_velo_2_cam"]
    y_true = item["y_true"][0]
    y_pred = item["y_pred"][0]
    id = item['id']

    # 调用 draw1 处理 pred 和 true
    draw1(id, "pred", y_pred, T_velo_2_cam, vox_origin, img_size, f, voxel_size, d)
    draw1(id, "true", y_true, T_velo_2_cam, vox_origin, img_size, f, voxel_size, d)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qt-plugin', default=None, help='Qt platform plugin')
    args = parser.parse_args()
    if args.qt_plugin:
        os.environ["QT_QPA_PLATFORM"] = args.qt_plugin

    scan = '/root/VoxFormer/S_16epoch.pkl'
    with open(scan, "rb") as handle:
        b = pickle.load(handle)

    print("len B:", len(b))
    vox_origin = np.array([0, -25.6, -2])
    img_size = (1220, 370)
    f = 707.0912

    # 使用多进程池
    num_processes = min(cpu_count(), len(b))  # 不超过 CPU 核心数或数据量
    print(f"Using {num_processes} processes")
    
    with Pool(processes=num_processes) as pool:
        # 使用 functools.partial 传递静态参数
        process_func = functools.partial(process_item, vox_origin=vox_origin, img_size=img_size, f=f)
        pool.map(process_func, b)

if __name__ == "__main__":
    main()