import urllib.request

url = "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
with urllib.request.urlopen(url) as response:
    content = response.read().decode('utf-8')

# Implement the requested sed replacement exactly: s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g
content = content.replace("deb https://", "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://")

with open("/home/abhinav/Data_Drive/roboracer/nvidia-container-toolkit.list", "w") as f:
    f.write(content)
