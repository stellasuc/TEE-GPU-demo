# TEE-GPU masked matmul prototype

这是一个“无真实 TEE”的 PyTorch 简易版实现，用来验证文档中的矩阵乘法掩码卸载思路。

代码把逻辑拆成两侧：

- trusted side：默认用 CPU 模拟 TEE，生成随机掩码、保存低秩因子、做 softmax 和校正恢复。
- untrusted GPU side：默认用 CUDA GPU，只接收 masked tensor，并执行大矩阵乘法。

注意：这个原型只验证算法路径，不提供真实安全边界。没有 TEE 时，进程内存、CUDA 显存、驱动和调试接口都不是可信隔离。

## 文件结构

- `tee_gpu_demo/masked_ops.py`：核心 PyTorch 算子。
- `tee_gpu_demo/llama_patch.py`：把 Llama 的 `nn.Linear` 替换成 masked 版本，并把 `LlamaAttention.forward` 改为 masked QK / trusted softmax / masked PV。
- `cache_llama.py`：预下载并缓存 Hugging Face 上的 Llama 模型。
- `demo_llama.py`：加载 Llama，把线性层替换成 masked 版本，并完成文本生成。
- `eval_accuracy.py`：在经典多选数据集上比较 baseline 和 masked 准确率。
- `verify_correctness.py`：用随机张量比较 baseline 与 masked/offload 路径的数值正确性。
- `profile_runtime.py`：用随机张量 profile baseline 与 masked/offload 路径的延迟、吞吐和显存峰值。
- `bench_ops.py`：测试 masked QK 的误差和耗时。
- `demo_kv_cache.py`：演示动态 masked K cache 的追加和查询。
- `demo_attention_cache.py`：演示 masked QK、TEE softmax、masked PV 的完整 attention 输出。
- `demo_continuous_batching.py`：演示请求到达、分块 prefill、连续 decode batching 的调度流程。
- `tests/test_masked_ops.py`：核心公式正确性测试。


## 安装

建议新建虚拟环境后安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3090 建议按 PyTorch 官网给出的 CUDA wheel 安装命令安装 `torch`。如果要使用 Meta Llama，需要先在 HuggingFace 上同意对应模型条款，并完成登录。

## 缓存 Llama

推理默认会优先读取本机已经下载好的模型目录：

```text
~/models/Llama-3.2-1B
```

代码会在启动时把它解析成绝对路径；如果该目录存在，`demo_llama.py` 和 `eval_accuracy.py` 会直接把这个本地目录传给 HuggingFace `from_pretrained()`。如果目录不存在，才回退到 `meta-llama/Llama-3.2-1B-Instruct` 和 `./model_cache` 的缓存流程。

先把模型下载到默认本地缓存目录 `./model_cache`：

```bash
python cache_llama.py
```

也可以用推理脚本只做下载：

```bash
python demo_llama.py --download-only
```

后续推理默认只读 `./model_cache`，不会再联网下载。默认模型和 trusted side 都在 CPU，GPU 只作为不可信 masked matmul 加速器：

```bash
python demo_llama.py --untrusted-device cuda --prompt "请用一句话解释什么是隐私保护矩阵乘法。"
```

`demo_llama.py` 默认使用 `--trusted-device cpu` 模拟“模型在 TEE 内”，并用 `--untrusted-device cuda` 执行 masked matmul。这个模式会有大量 CPU/GPU 传输，主要用于验证方案边界，不代表最终性能。

如果只是为了兼容“模型整体放 GPU”的 HuggingFace demo，可以把 trusted device 也显式设成 CUDA，并打开兼容模式：

```bash
python demo_llama.py --trusted-device cuda --untrusted-device cuda --compat-return-to-model-device --prompt "Hello"
```

如果要使用其他缓存目录，下载和推理时传同一个 `--cache-dir`：

```bash
python cache_llama.py --cache-dir /path/to/model_cache
python demo_llama.py --untrusted-device cuda --cache-dir /path/to/model_cache --prompt "Hello"
```

推理阶段默认本地只读。如果确实希望推理脚本在缓存缺失时联网补下载，需要显式加 `--allow-download`：

```bash
python demo_llama.py --untrusted-device cuda --allow-download --prompt "Hello"
```

## 运行测试

```bash
python -m unittest discover -s tests
```

独立正确性验证，不需要下载模型：

```bash
python verify_correctness.py --untrusted-device cpu
python verify_correctness.py --trusted-device cpu --untrusted-device cuda
```

脚本会比较 masked Linear、masked QK/PV、动态 K/V cache、完整 masked attention cache，以及被 patch 后的 LlamaAttention-style prefill/decode cache 输出。

## Profiling

不加载 Hugging Face 模型，直接 profile 随机张量路径：

```bash
python profile_runtime.py --target all --untrusted-device cpu
python profile_runtime.py --target attention --trusted-device cpu --untrusted-device cuda
python profile_runtime.py --target llama --trusted-device cpu --untrusted-device cuda
python profile_runtime.py --target continuous --batch 8 --trusted-device cpu --untrusted-device cuda
```

可选导出 PyTorch profiler Chrome trace：

```bash
python profile_runtime.py --target llama --torch-profiler --trace-file trace.json
```

`--target` 支持 `linear`、`qk`、`pv`、`kv`、`attention`、`llama`、`continuous` 和 `all`。
其中 `--target llama` 会同时输出完整 prompt prefill 和单 token decode step 的 baseline/masked 结果；decode 会先构建 1 次 prefill cache，再在计时区间连续执行 `--repeats` 个 decode token。
输出里的 `speedup=1.20x` 表示 masked/offload 路径比 baseline 快 20%；小于 `1.0x` 表示当前配置下反而更慢，后面的 `overhead` 会显示额外开销比例。

## 准确率评估

`eval_accuracy.py` 会先跑未替换的 baseline，再原地 patch 为 masked Linear 后跑同一批样本，最后输出准确率和 `masked - baseline` 差值。
masked run 默认使用 `--trusted-device cpu` 模拟 TEE。

首次运行某个数据集时，需要允许下载一次数据集缓存：

```bash
python eval_accuracy.py --untrusted-device cuda --tasks piqa,arc_easy,boolq --limit 100 --allow-dataset-download
```

后续默认只读本地 `./model_cache` 和 `./dataset_cache`，不会联网：

```bash
python eval_accuracy.py --untrusted-device cuda --tasks piqa,arc_easy,boolq --limit 100
```

支持的数据集：

```text
piqa, arc_easy, arc_challenge, hellaswag, winogrande, boolq
```

评估更多任务并保存结果：

```bash
python eval_accuracy.py \
  --untrusted-device cuda \
  --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,boolq \
  --limit 200 \
  --output-json eval_results.json
```

如果模型还没有缓存，先运行 `python cache_llama.py`；如果你确实希望评估脚本自动下载模型，需要显式加 `--allow-model-download`。

## 算子 benchmark

```bash
python bench_ops.py --device cuda --m 128 --n 1024 --d 128 --rank 4
```

## 动态 KV cache demo

```bash
python demo_kv_cache.py --device cuda --tokens 1024 --chunk 128 --dim 128
```

等价于：

```bash
python demo_kv_cache.py --trusted-device cpu --untrusted-device cuda --tokens 1024 --chunk 128 --dim 128
```

## Masked PV attention demo

这个 demo 实现：

```text
GPU: masked QK
TEE: score correction + softmax
GPU: masked P @ masked V
TEE: output correction
```

运行：

```bash
python demo_attention_cache.py --device cuda --tokens 1024 --chunk 128 --dim 128
```

其中真实 Q/K/V、softmax、校正项都在 CPU；GPU 只看到 masked Q/K/P/V 并执行 masked QK 和 masked PV。

## Continuous batching demo

这个 demo 用随机 per-head Q/K/V 模拟多请求服务调度：请求可以按不同 step 到达，每个请求独立维护 masked K/V cache，调度器每轮接收新请求、分块 prefill，并从活跃请求中组成最多 `--max-batch-size` 个 decode token 的连续 batch。同一轮 decode 会调用 `batched_masked_attention_query`，把不同请求的 masked QK 和 masked `P @ V` padding 后合并成 batched GPU matmul；trusted side 仍逐请求做 correction 和 softmax。

```bash
python demo_continuous_batching.py --trusted-device cpu --untrusted-device cuda --requests 8 --max-batch-size 4
```

在没有 CUDA 的环境可以先用 CPU 跑：

```bash
python demo_continuous_batching.py --untrusted-device cpu --timeline
```

输出里的 `step_reduction` 是相对逐请求顺序执行的调度步数压缩比例；`max_abs_error` / `mean_abs_error` 会把连续 batching 输出和 plain attention reference 对齐，验证调度不改变结果。

## Llama demo

默认使用 `meta-llama/Llama-3.2-1B-Instruct`。运行一次推理：

```bash
python demo_llama.py --untrusted-device cuda --prompt "请用一句话解释什么是隐私保护矩阵乘法。"
```

不传 `--prompt` 时，脚本会在终端里询问一次 prompt。

交互式连续输入 prompt：

```bash
python demo_llama.py --untrusted-device cuda --interactive
```

只替换注意力投影层：

```bash
python demo_llama.py --untrusted-device cuda --layers q_proj,k_proj,v_proj,o_proj
```

只验证 masked Linear、暂时不替换 `LlamaAttention.forward`：

```bash
python demo_llama.py --untrusted-device cuda --disable-masked-attention --prompt "Hello"
```

调试时对比替换前后的 logits 误差：

```bash
python demo_llama.py --untrusted-device cuda --prompt "Hello" --compare-baseline
```

显存紧张时，优先使用 1B 模型；3B 可以尝试，但 10GB 显存下余量会更小。

当前 Llama 推理路径会把选中的 `nn.Linear` 层替换为 masked Linear，并默认重写 HuggingFace `LlamaAttention.forward`：GQA 下按 KV head 维护 masked K/V cache，GPU 侧 masked K/V 使用 append-only contiguous buffer 复用历史 token，同一 KV group 内的多个 query heads 会合并为一次 masked attention query，把 masked QK 与 masked `P @ V` 外包给 `--untrusted-device`，score correction、softmax、输出 correction 留在 `--trusted-device`。这个实现优先验证安全边界和公式正确性，会按 batch/KV group 做 Python 级循环，性能不代表最终工程化版本。

masked Linear、masked QK、masked `P @ V` 都采用 launch-then-correct 的流水线顺序：先发起 untrusted/GPU matmul，再在 trusted side 计算 correction，最后等待 GPU 结果并合并；CUDA 场景下 correction 有机会和 GPU matmul 重叠。
