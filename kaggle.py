import kagglehub

# download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("path to dataset files:", path)