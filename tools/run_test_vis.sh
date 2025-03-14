#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供 vis_id 的值作为参数。"
    exit 1
fi

vis_id=$1
python_file="projects/configs/voxformer/voxformer-S.py"

echo "当前 vis_id 的值为: $vis_id"

# 修改 Python 文件中的 vis_id 参数
sed -i "s/^\(vis_id *= *\).*/\1'$vis_id'/" $python_file
# 执行第一条命令
tools/dist_test_vis.sh projects/configs/voxformer/voxformer-S.py /root/VoxFormer/result/voxformer-S/epoch_6.pth 1

# 执行第二条命令
xvfb-run -a python tools/vis_pkl.py
    