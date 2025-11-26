#pip install requests
import requests,json

url='http://localhost:5000/predict'

# Set image file path
image_path='data/infer_data/IDP01310622.jpg'

# Read image file and set as payload
image=open(image_path, 'rb')
payload= {'image': image}
# send_content = json.dumps(payload)

headers={'Accept': 'application/json, text/plain, */*'}
# Send POST request with image and get response
response=requests.post(url, files=payload)

print(response.text)