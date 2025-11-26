# import cv2

# img=cv2.imread("image_endplate.png")
# print()

import json,cv2
import base64
import numpy as np
json_file=r"C:\Users\Administrator\Desktop\json_str.log"

with open(json_file,"r") as f:
    json_str=f.readlines()

json_obj=json.loads(json_str[0])
images_b64=json_obj['ret']['images_drawed'][0]

image_b64=images_b64[1]
image_data=base64.b64decode(image_b64)
image_arr=np.frombuffer(image_data,np.uint8)
image=cv2.imdecode(image_arr, cv2.COLOR_RGB2BGR)


print(json_str)