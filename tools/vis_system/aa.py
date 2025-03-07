from flask import Flask, request, jsonify, send_from_directory
import base64
from io import BytesIO
from PIL import Image, ImageOps

app = Flask(__name__)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'a.html')  # 假设 `a.html` 在同一目录下

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    if 'image' not in data or 'model' not in data:
        return jsonify({'message': '缺少必要数据'}), 400

    # 解码图像
    image_data = data['image'].split(',')[1]  # 去掉base64头部
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # 根据选择的模型进行图像处理
    model = data['model']
    if model == 'grayscale':
        processed_image = ImageOps.grayscale(image)
    elif model == 'invert':
        processed_image = ImageOps.invert(image.convert("RGB"))  # 转换为 RGB 后反色
    else:
        return jsonify({'message': '不支持的模型类型'}), 400

    # 编码处理后的图像
    buffered = BytesIO()
    processed_image.save(buffered, format="PNG")
    processed_image_str = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        'message': '图像处理成功',
        'processed_image': processed_image_str
    })

if __name__ == '__main__':
    app.run(debug=True)
