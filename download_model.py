from modelscope import snapshot_download
model_dir = snapshot_download('ckdckd/OpticsGPT-v0')
print(f"模型已下载至: {model_dir}")