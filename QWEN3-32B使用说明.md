# Qwen3-32B 本地模型配置说明

本文档说明如何使用本地 Qwen3-32B 模型运行四种方法：MRAG、zero_shot_cot、rag_cot 和 react。

## 前置准备

### 1. 启动 VLLM 服务

在运行任何方法之前，需要先启动本地 VLLM 服务：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3-32B \
    --served-model-name qwen3-32b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --disable-log-requests
```

**参数说明：**
- `--model`: 本地模型路径
- `--served-model-name`: 服务中使用的模型名称（必须是 `qwen3-32b`）
- `--port`: 服务端口（默认 8000）
- `--tensor-parallel-size`: 张量并行大小，根据 GPU 数量调整
- `--max-model-len`: 最大上下文长度

**注意：** 启动服务后，保持该终端运行，在另一个终端中运行下面的方法。

### 2. 验证服务是否启动成功

```bash
curl http://localhost:8000/v1/models
```

应该看到返回包含 `qwen3-32b` 模型的 JSON 响应。

## 运行方法

### 方法 1-3：timeqa_baseline_lab (zero_shot_cot, rag_cot, react)

这三个方法都在 `timeqa_baseline_lab` 项目中，已经配置为默认使用 Qwen3-32B。

#### 配置文件

默认配置在 [configs/default.yaml](timeqa_baseline_lab/configs/default.yaml):

```yaml
model:
  provider: "vllm"              # 使用 VLLM provider
  model_name: "qwen3-32b"       # 模型名称（必须与 VLLM 服务中的名称一致）
  base_url: "http://localhost:8000/v1"
  max_new_tokens: 256
  temperature: 0.0              # 0.0 表示贪婪解码（确定性输出）
```

**关键配置：**
- `provider: "vllm"`: 使用 VLLM provider（通过 OpenAI 兼容 API）
- `model_name: "qwen3-32b"`: 必须与 VLLM 服务的 `--served-model-name` 一致
- 思考功能已通过 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关闭

#### 运行示例

进入项目目录：
```bash
cd /workspace/ETE-Graph/agent/timeqa_baseline_lab
```

**运行 zero_shot_cot：**
```bash
python -m timeqa_baseline_lab.run \
    --config configs/default.yaml \
    --strategy zero_shot_cot
```

**运行 rag_cot：**
```bash
python -m timeqa_baseline_lab.run \
    --config configs/default.yaml \
    --strategy rag_cot
```

**运行 react：**
```bash
python -m timeqa_baseline_lab.run \
    --config configs/default.yaml \
    --strategy react
```

### 方法 4：MRAG

MRAG 方法已经修改为支持 Qwen3-32B（通过 `qwen3_32b` reader 选项）。

#### 运行示例

进入 MRAG 目录：
```bash
cd /workspace/ETE-Graph/agent/MRAG-master
```

运行 MRAG：
```bash
python reader.py \
    --reader qwen3_32b \
    --retriever_output <检索结果文件路径> \
    --ctx_topk 5 \
    --paradigm concat
```

**参数说明：**
- `--reader qwen3_32b`: 使用 Qwen3-32B 模型（已配置关闭思考功能）
- `--retriever_output`: 检索结果文件路径
- `--ctx_topk`: 使用的上下文数量
- `--paradigm`: 使用的范式（concat 或 fusion）

## 代码修改总结

### 1. timeqa_baseline_lab 修改

**文件：** `src/timeqa_baseline_lab/llm.py`
- 添加了 `VLLMGenerator` 类，用于连接本地 VLLM 服务
- 在 `build_generator` 函数中添加了对 `vllm` provider 的支持
- 自动添加 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关闭思考功能

**文件：** `src/timeqa_baseline_lab/config.py`
- 更新 `ModelConfig` 默认值为 `vllm` provider
- 在 `_build_experiment` 中添加对 `vllm` provider 的验证

**文件：** `configs/default.yaml`
- 更新默认配置为使用 VLLM 和 Qwen3-32B

### 2. MRAG 修改

**文件：** `utils.py`
- 在 `llm_names` 函数中添加了 `qwen3_32b` 选项
- 在 `call_pipeline` 函数中添加了对 `qwen3_32b` 的处理逻辑
- 使用 OpenAI 客户端连接本地 VLLM 服务
- 自动添加 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关闭思考功能

**文件：** `reader.py`
- 添加 `qwen3_32b` 到 reader 选项
- 更新默认 reader 为 `qwen3_32b`
- 修改逻辑使 `qwen3_32b` 不加载 VLLM 库（使用 OpenAI API 代替）

## 常见问题

### Q1: 如何验证思考功能已关闭？

在代码中，我们已经通过以下方式关闭了思考功能：
```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

这个参数会传递给 VLLM 服务，确保模型不会生成思考过程。

### Q2: 如何修改生成参数？

**timeqa_baseline_lab：** 修改 `configs/default.yaml` 中的参数：
```yaml
model:
  max_new_tokens: 512        # 增加最大生成长度
  temperature: 0.7           # 增加随机性
```

**MRAG：** 在 `utils.py` 的 `call_pipeline` 函数中修改：
```python
temperature=0.7,           # 当前是 0.2
max_tokens=max_tokens,     # 通过函数参数传递
```

### Q3: 如何使用其他模型？

1. 修改 VLLM 启动命令中的 `--model` 路径
2. 保持 `--served-model-name qwen3-32b` 不变（或者修改所有配置文件中的模型名称）
3. 重启 VLLM 服务

### Q4: VLLM 服务占用内存过大怎么办？

可以通过以下方式减少内存占用：
- 减少 `--max-model-len` 参数
- 使用量化版本的模型
- 减少 `--tensor-parallel-size`（但会降低推理速度）

### Q5: 遇到连接错误怎么办？

检查以下几点：
1. VLLM 服务是否正常运行（检查端口 8000）
2. 配置文件中的 `base_url` 是否正确（`http://localhost:8000/v1`）
3. 防火墙是否阻止了连接

## 技术细节

### 思考功能关闭原理

Qwen3 系列模型支持思考模式（thinking mode），在该模式下模型会先生成思考过程再给出答案。通过设置：
```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

我们告诉 VLLM 服务在处理 chat template 时不启用思考模式，直接生成答案。

### VLLM vs HuggingFace Transformers

**VLLM 优势：**
- 更高的推理速度（PagedAttention 优化）
- 更好的批处理性能
- 更低的延迟
- 支持多 GPU 推理

**HuggingFace Transformers：**
- 更简单的安装和使用
- 不需要启动独立服务
- 适合小规模测试

## 性能优化建议

1. **批处理：** 对于 timeqa_baseline_lab，可以在配置文件中增加 `batch_size`（仅适用于非交互式策略）
2. **并行度：** 根据 GPU 数量调整 `--tensor-parallel-size`
3. **上下文长度：** 根据实际需求调整 `--max-model-len`，避免浪费显存
4. **温度参数：** 对于 QA 任务，建议使用 `temperature: 0.0` 以获得确定性输出

## 联系方式

如有问题，请参考：
- VLLM 文档：https://docs.vllm.ai/
- Qwen 模型文档：https://github.com/QwenLM/Qwen
