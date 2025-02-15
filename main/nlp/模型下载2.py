from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 配置参数
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 确认模型名称是否正确
SAVE_PATH = "/nlp/models/Qwen2.5-0.5B-Instruct"  # 本地保存路径

# 创建保存目录
os.makedirs(SAVE_PATH, exist_ok=True)

try:
    # 修改后的下载代码（移除了resume_download参数）
    print("正在下载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # resume_download=True  # 新版本已默认启用断点续传
    )
    model.save_pretrained(SAVE_PATH)
    
    # 下载并保存tokenizer
    print("正在下载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(SAVE_PATH)
    
    print(f"模型和分词器已成功保存至：{SAVE_PATH}")

except Exception as e:
    print(f"下载失败，错误信息：{str(e)}")
