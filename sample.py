"""
Sample from a trained model
"""
import os
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model1 import GPTConfig, GPT
from torch.nn import functional as F
from data_deal import load_and_preprocess_data, split_data

# 配置和初始化
init_from = 'resume'
out_dir = 'out'
seed = 1337
device = 'cuda'
dtype = 'float64' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
# ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float64': torch.float64  # 或 torch.double
}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 模型加载
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# 示例数据加载
file_path = 'D:/yansdnu/traffic forecast/dataset/milano_traffic_nid.csv'
window_size = 3

# 加载并预处理数据
x, y = load_and_preprocess_data(file_path, window_size)

# 分割数据
x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

# 定义数据加载器
train_data = (x_train, y_train)
val_data = (x_val, y_val)
test_data = (x_test, y_test)

def get_batch(split: str, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    if split == 'train':
        data = train_data
    elif split == 'val':
        data = val_data
    elif split == 'test':
        data = test_data
    else:
        raise ValueError("Invalid split name. Use 'train', 'val' or 'test'.")

    x_data, y_data = data
    ix = torch.randint(len(x_data), (batch_size,))
    x_batch = torch.stack([torch.from_numpy(np.array(x_data[i])).float() for i in ix])
    y_batch = torch.tensor([y_data[i] for i in ix], dtype=torch.float32)
    x_batch = x_batch.unsqueeze(-1)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    return x_batch, y_batch


# 获取测试批次
batch_size = 200
x_test_batch, y_test_batch = get_batch('test', batch_size)

# 进行预测并计算MSE
with torch.no_grad():
    predictions = model(x_test_batch)[0].squeeze(-1)
    mse = F.mse_loss(predictions, y_test_batch)
    print(f'均方误差: {mse.item()}')
    print("预测值与实际值:")
    for pred, actual in zip(predictions, y_test_batch):
        print(f'预测: {pred.item()}, 实际: {actual.item()}')