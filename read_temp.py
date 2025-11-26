import os
import imghdr
from datetime import datetime

directory_path = "/root/WORKSPACE/workspce/BSpine-flask/temp"

file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

print("=" * 40)

file_info_list = []

for file_name in os.listdir(directory_path):
    file_path = os.path.join(directory_path, file_name)
    if os.path.isfile(file_path):
        file_stats = os.stat(file_path)
        creation_time = datetime.fromtimestamp(file_stats.st_ctime)
        file_info_list.append((file_name, creation_time))

file_info_list.sort(key=lambda x: x[1])

for file_name, creation_time in file_info_list:
    print(f"文件名: {file_name}, 创建时间: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")

latest_file_name, latest_creation_time = file_info_list[-1]
latest_file_path = os.path.join(directory_path, latest_file_name)

print(f"最新文件: {latest_file_name}, 创建时间: {latest_creation_time.strftime('%Y-%m-%d %H:%M:%S')}")

with open(latest_file_path, "rb") as file:
    content = file.read()

file_type = imghdr.what(None, content)
if file_type:
    print(f"检测到的文件类型: {file_type}")
    output_image_path = "output.jpeg"
    with open(output_image_path, "wb") as img_file:
        img_file.write(content)
    print(f"图片已保存为: {output_image_path}")
else:
    print(f"最新文件 '{latest_file_name}' 不是图片类型。")

print(f"当前文件夹 '{directory_path}' 中的文件总数: {file_count}")