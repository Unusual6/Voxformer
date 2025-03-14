# 定义输入文件路径
input_file = '/root/VoxFormer/id_list.txt'
# 定义输出文件路径
output_file = 'id.txt'

try:
    # 以只读模式打开输入文件
    with open(input_file, 'r') as f:
        # 读取文件中的所有行
        lines = f.readlines()
        # 去除每行末尾的换行符，并将其转换为整数
        numbers = [int(line.strip()) for line in lines]

    # 对数字列表进行排序
    sorted_numbers = sorted(numbers)

    # 以写入模式打开输出文件
    with open(output_file, 'w') as f:
        # 遍历排序后的数字列表
        for number in sorted_numbers:
            # 将数字格式化为6位字符串，不足6位时在前面补0
            formatted_number = '{:06d}'.format(number)
            # 将格式化后的数字写入文件，并添加换行符
            f.write(formatted_number + '\n')

    print(f"排序完成，结果已保存到 {output_file}")
except FileNotFoundError:
    print(f"错误：未找到文件 {input_file}")
except Exception as e:
    print(f"发生未知错误：{e}")
    