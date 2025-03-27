from flask import Flask, request, jsonify, send_from_directory
import base64 , cv2 ,os
from io import BytesIO
from PIL import Image, ImageOps
import subprocess

app = Flask(__name__)

@app.route('/')
def serve_html():
    return send_from_directory('.', './static/main.html')  # 假设 `a.html` 在同一目录下

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json

    if 'image' not in data or 'model' not in data:
        return jsonify({'message': '缺少必要数据'}), 400
    print("image_name:",data["image_name"])
    # 解码图像
    image_data = data['image'].split(',')[1]  # 去掉base64头部
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    #执行
    id=data["image_name"].split('.')[0]
    # id = '000900'
    print("id:",id)
    output_path = "/root/VoxFormer/vis_output"
    # subprocess.run(["pkill", "-f", "Xvfb"], check=False)  # 清理残留 Xvfb
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # env["DISPLAY"] = ":99"  # 可选，留给 xvfb-run 自动分配
    model = data['model']


    
    # print("PLY Data (Backend):", ply_data[:100]) 
    subprocess.run(["sh","/root/VoxFormer/tools/run_test_vis.sh",id ,model], cwd="/root/VoxFormer",
        # capture_output=True,
        text=True,
        env={"PYTHONUNBUFFERED": "1"}  # 确保实时输出
        )


    image_path = os.path.join(output_path, id + '_pred_ssc.png') 
    if not os.path.exists(image_path):
        return jsonify({'error': '预测图像未找到'})

    with open(image_path, "rb") as image_file:
        pred_image_str = base64.b64encode(image_file.read()).decode()

    ply_path='/root/VoxFormer/voxel_grid1.ply'
    try:
        with open(ply_path, "rb") as ply_file:
            ply_binary = ply_file.read()
            # print("PLY 文件大小:", len(ply_binary))  # 🔍 确保文件不是空的
            ply_data = base64.b64encode(ply_binary).decode()
    except Exception as e:
        print("Base64 编码错误:", str(e))
        
    return jsonify({
        'message': '图像处理成功',
        'processed_image': pred_image_str,
        'ply_data': ply_data
    })

if __name__ == '__main__':
    app.run(debug=True)
