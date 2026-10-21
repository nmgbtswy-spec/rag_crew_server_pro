from modelscope import snapshot_download

print("开始从镜像源下载模型...")
model_dir = snapshot_download(
    'AI-ModelScope/beg-small-zh-v1.5',
    local_dir='./.bge_model'
)
print(f"下载完成，模型保存在：{model_dir}")
