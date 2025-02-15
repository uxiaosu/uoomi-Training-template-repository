import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np

class DistillationTrainer:
    def __init__(self, config):
        # 初始化配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model = None
        self.student_model = None
        self.optimizer = None
        self.scheduler = None
        self.tokenizer = None
        self.config = config
        
        # 加载模型和分词器
        self._load_models()
        
    def _load_models(self):
        """加载教师和学生模型"""
        # 教师模型（大模型）
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.config['teacher_model'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()  # 教师模型设为评估模式
        
        # 学生模型（小模型）
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.config['student_model'],
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        ).train()
        
        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['teacher_model'],
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _prepare_data(self):
        """准备训练数据"""
        dataset = load_dataset(
            self.config['dataset_name'],
            self.config['dataset_config'],
            split=f"train[:{self.config['max_samples']}%]"
        )
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=self.config['max_length'],
                return_tensors="pt"  # 确保返回PyTorch张量
            )
            
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
        # 设置数据集格式为PyTorch张量
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        return DataLoader(
            tokenized_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=lambda batch: {
                'input_ids': torch.stack([x['input_ids'] for x in batch]),
                'attention_mask': torch.stack([x['attention_mask'] for x in batch])
            }
        )
        
    def _compute_loss(self, student_outputs, teacher_outputs, labels):
        """计算蒸馏损失"""
        # 知识蒸馏损失（KL散度）
        temperature = self.config['temperature']
        loss_kl = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(student_outputs.logits / temperature, dim=-1),
            nn.functional.softmax(teacher_outputs.logits.detach() / temperature, dim=-1)
        ) * (temperature ** 2)
        
        # 学生模型的标准交叉熵损失
        loss_ce = nn.CrossEntropyLoss()(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            labels.view(-1)
        )
        
        return self.config['alpha'] * loss_kl + (1 - self.config['alpha']) * loss_ce
        
    def train(self):
        """训练循环"""
        dataloader = self._prepare_data()
        
        # 优化器和学习率调度
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=len(dataloader) * self.config['epochs']
        )
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            total_loss = 0
            self.student_model.train()
            
            for step, batch in enumerate(dataloader):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                labels = inputs['input_ids']
                
                # 教师模型推理
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                
                # 学生模型推理
                student_outputs = self.student_model(**inputs)
                
                # 计算损失
                loss = self._compute_loss(student_outputs, teacher_outputs, labels)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                
                # 打印进度
                if self.config['verbose'] and (step % 10 == 0):
                    print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
            
        # 保存学生模型
        self.student_model.save_pretrained(self.config['save_dir'])
        self.tokenizer.save_pretrained(self.config['save_dir'])

if __name__ == "__main__":
    # 配置参数
    config = {
        'teacher_model': "nlp/models/DeepSeek-R1-Distill-Qwen-1.5B",  # 使用正斜杠
        'student_model': "nlp/models/Qwen2.5-0.5B-Instruct",
        'dataset_name': "wikitext",
        'dataset_config': "wikitext-2-raw-v1",  # 新增配置项
        'max_samples': 10,
        'max_length': 512,
        'batch_size': 2,
        'epochs': 3,
        'learning_rate': 5e-5,
        'temperature': 2.0,
        'alpha': 0.7,  # KL散度损失权重
        'save_dir': "./distilled_model",
        'verbose': True
    }
    
    # 初始化并训练
    trainer = DistillationTrainer(config)
    trainer.train() 