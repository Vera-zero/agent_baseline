#!/bin/bash

# Qwen3-32B 配置验证脚本

echo "================================"
echo "Qwen3-32B 配置验证"
echo "================================"
echo ""

# 检查 1: VLLM 服务是否运行
echo "[1] 检查 VLLM 服务..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✓ VLLM 服务正常运行"
    curl -s http://localhost:8000/v1/models | python -m json.tool | grep -A 2 "id"
else
    echo "✗ VLLM 服务未运行"
    echo "  请先启动 VLLM 服务："
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "      --model /workspace/models/Qwen3-32B \\"
    echo "      --served-model-name qwen3-32b \\"
    echo "      --host 0.0.0.0 --port 8000 \\"
    echo "      --tensor-parallel-size 2 \\"
    echo "      --max-model-len 32768"
    exit 1
fi
echo ""

# 检查 2: timeqa_baseline_lab 配置
echo "[2] 检查 timeqa_baseline_lab 配置..."
if [ -f "/workspace/ETE-Graph/agent/timeqa_baseline_lab/configs/default.yaml" ]; then
    echo "✓ 配置文件存在"
    echo "  Provider: $(grep -A 1 'provider:' /workspace/ETE-Graph/agent/timeqa_baseline_lab/configs/default.yaml | head -1 | awk '{print $2}')"
    echo "  Model: $(grep 'model_name:' /workspace/ETE-Graph/agent/timeqa_baseline_lab/configs/default.yaml | head -1 | awk '{print $2}')"
else
    echo "✗ 配置文件不存在"
    exit 1
fi
echo ""

# 检查 3: MRAG 代码修改
echo "[3] 检查 MRAG 代码修改..."
if grep -q "qwen3_32b" /workspace/ETE-Graph/agent/MRAG-master/utils.py; then
    echo "✓ MRAG utils.py 已修改（包含 qwen3_32b）"
else
    echo "✗ MRAG utils.py 未正确修改"
    exit 1
fi

if grep -q "qwen3_32b" /workspace/ETE-Graph/agent/MRAG-master/reader.py; then
    echo "✓ MRAG reader.py 已修改（包含 qwen3_32b）"
else
    echo "✗ MRAG reader.py 未正确修改"
    exit 1
fi
echo ""

# 检查 4: Python 依赖
echo "[4] 检查 Python 依赖..."
python -c "import openai; print('✓ openai 已安装')" 2>/dev/null || echo "✗ openai 未安装（需要运行: pip install openai）"
python -c "import vllm; print('✓ vllm 已安装')" 2>/dev/null || echo "⚠ vllm 未安装（MRAG 可能需要）"
python -c "import transformers; print('✓ transformers 已安装')" 2>/dev/null || echo "✗ transformers 未安装"
echo ""

# 检查 5: 测试简单调用
echo "[5] 测试 VLLM 服务调用..."
cat > /tmp/test_vllm.py << 'EOF'
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

try:
    response = client.chat.completions.create(
        model="qwen3-32b",
        messages=[
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
        ],
        max_tokens=50,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("✓ VLLM 服务调用成功")
    print(f"  响应: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ VLLM 服务调用失败: {e}")
EOF

python /tmp/test_vllm.py
echo ""

echo "================================"
echo "验证完成！"
echo "================================"
echo ""
echo "如果所有检查都通过，可以开始使用修改后的代码。"
echo "详细使用说明请参考: /workspace/ETE-Graph/agent/QWEN3-32B使用说明.md"
