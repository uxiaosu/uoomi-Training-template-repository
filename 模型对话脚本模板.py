from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
import threading
import time

class ChatAssistant:
    def __init__(self, model_path="nlp\models\Qwen2.5-0.5B-Instruct"):
        # 初始化模型和分词器
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        self.history = []
        print(f"✅ 模型已加载到 {self.device.upper()} 设备")

    def generate_stream(self, query, max_length=512, temperature=0.7, top_p=0.9):
        # 构建输入
        input_ids = self.tokenizer.encode(
            f"<|User|>: {query}\n<|Bot|>:",
            return_tensors="pt"
        ).to(self.device)
        
        # 流式输出设置
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # 启动生成线程
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 流式输出结果
        generated_text = ""
        print("🤖: ", end="", flush=True)
        for new_text in streamer:
            generated_text += new_text
            print(new_text, end="", flush=True)
            yield generated_text
        
        # 保存对话历史
        self.history.append((query, generated_text))
        if len(self.history) > 5:  # 保留最近5轮对话
            self.history.pop(0)

    def chat_loop(self):
        print("欢迎使用DeepSeek对话助手！输入'exit'退出")
        while True:
            try:
                user_input = input("\n👤: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                start_time = time.time()
                print("生成中...", end="\r")
                
                # 流式生成
                for response in self.generate_stream(user_input):
                    pass  # 实时输出已通过流式处理
                
                print(f"\n⏱️ 响应时间：{time.time()-start_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n🛑 用户中断")
                break
            except Exception as e:
                print(f"\n❌ 生成错误：{str(e)}")
                continue

if __name__ == "__main__":
    # 初始化助手
    assistant = ChatAssistant()
    
    # 启动对话
    assistant.chat_loop() 