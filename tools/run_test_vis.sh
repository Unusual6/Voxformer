#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供 vis_id 的值作为参数。"
    exit 1
fi

vis_id=$1

model=$2


# 去除 model 变量的前后空格，防止匹配失败
model=$(echo "$model" | xargs)

# 检查 model 是否为空
if [ -z "$model" ]; then
    echo "Error: model 变量为空！请传入参数！"
    exit 1
fi

if [ "$model" = "normal" ]; then
    echo "当前 model 的值为: $model"
    python_file="projects/configs/voxformer/voxformer-S.py"
    ckpts="/root/VoxFormer/result/voxformer-S/latest.pth"

elif [ "$model" = "shixu" ]; then
    echo "当前 model 的值为: $model"
    python_file="projects/configs/voxformer/voxformer-T_deform3D.py"
    ckpts="/root/VoxFormer/result/voxformer-T_deform3D/latest.pth"

elif [ "$model" = "zhou" ]; then
    echo "当前 model 的值为: $model"
    python_file="projects/configs/voxformer/voxformer-T_deform3D.py"
    ckpts="/root/VoxFormer/result/voxformer-T_deform3D/latest.pth"

else
    echo "Error: Unknown model type '$model'"
    exit 1
fi

echo "当前 vis_id 的值为: $vis_id"

# 修改 Python 文件中的 vis_id 参数
sed -i "s/^\(vis_id *= *\).*/\1'$vis_id'/" $python_file

# conda activate vox
# # 执行第一条命令
# # /opt/conda/envs/vox/bin/python 
# /opt/conda/bin/conda run -n vox tools/dist_test_vis.sh $python_file $ckpts 1
# echo "after test!!!"
# # 执行第二条命令
# /opt/conda/bin/conda run -n vox xvfb-run -a python tools/vis_pkl.py 
# echo "after vis!!!"

export PATH=/opt/conda/envs/vox/bin:$PATH
# 直接使用 Conda 解释器
PYTHON_EXEC=/opt/conda/envs/vox/bin/python

# 执行 dist_test_vis.sh，保证 `python` 解析正确
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
$PYTHON_EXEC -m torch.distributed.launch --nproc_per_node=1 --master_port=29503 \
    $(dirname "$0")/test.py "$python_file" "$ckpts"

echo "--------after test-----------"

# 运行可视化脚本
xvfb-run -a $PYTHON_EXEC tools/vis_pkl.py
echo "=========after vis=========="
