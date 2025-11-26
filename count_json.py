import os
import imghdr
from datetime import datetime

directory_path = "/root/WORKSPACE/workspce/BSpine-flask/data/images"

file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

print(f"当前文件夹 '{directory_path}' 中的文件总数: {file_count}")