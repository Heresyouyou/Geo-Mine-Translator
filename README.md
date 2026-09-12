# Geo-Mine-Translator

地学/矿业 PDF 本地离线翻译系统 — llama.cpp (GGUF) + BabelDOC v2 + Flask Web UI.
RTX 4060 8GB 流畅运行, 4 并发 GPU, 14000+ 地学术语.

**San Albino 16.1MB/33页: 254s / 190 segments / 0 errors / -26% vs 初始版本**

## 快速开始

`powershell
# 1. 启 llama-server
./llama-cpp/llama-server.exe -m ./models-gguf/Qwen3.5-4B-Q4_K_M.gguf 
  -ngl -1 -c 16384 -t 4 -b 2048 --parallel 4 --kv-unified 
  --cache-prompt --reasoning off --cache-idle-slots --host 127.0.0.1 --port 8090

# 2. 启 Flask
py -3.12 web_app.py

# 3. 浏览器打开 http://127.0.0.1:5555
`

或双击 start_translate.bat 一键启动.

## 系统要求

| 项目 | 要求 |
|------|------|
| GPU | RTX 30/40 系, >= 6GB VRAM (推荐 8GB) |
| CUDA | 12.x |
| Python | 3.12.x |
| 系统 | Windows 10/11 |
| 磁盘 | ~4GB (GGUF) |

### 核心依赖

`
BabelDOC==0.2.33
Flask==3.1.3
torch==2.5.1+cu121
requests==2.34.2
pillow pdfminer.six PyMuPDF pikepdf transformers
`

## 模型下载

| 模型 | 大小 | VRAM | 下载 |
|------|------|------|------|
| Qwen3.5-4B-Q4_K_M | 2.55 GB | ~4 GB | [HF](https://huggingface.co/Qwen/Qwen3.5-4B-Instruct-GGUF) / [ModelScope](https://modelscope.cn/models/qwen/Qwen3.5-4B-Instruct-GGUF) |
| Qwen3.5-0.8B-Q4_K_M | 0.5 GB | ~1.2 GB | 同上 (降级) |
| Hunyuan-MT-7B-Q4_K_M | 4.31 GB | ~6.5 GB | [HF](https://huggingface.co/moonshotai/Hunyuan-MT-7B-GGUF) |

国内直连: modelscope download --model qwen/Qwen3.5-4B-Instruct-GGUF --local_dir ./models-gguf --allow_patterns "*Q4_K_M.gguf"

## 关键配置

### run_babeldoc.py (BabelDOC v2 API)

`python
set_translate_rate_limiter(8)      # BabelDOC QPS 8, 每 125ms 放 1 段
TranslationConfig(
    qps=16,                        # max_workers = min(16*2, 16+5) = 21 线程
    disable_rich_text_translate=True,  # 不拆蓝色超链接/彩色文字
    skip_scanned_detection=True,   # 论文都是数字 PDF
    table_model=None,              # 关 OCR (不需要)
    watermark_output_mode=NoWatermark,
    enhance_compatibility=False,
)
`

### llama_http_engine.py

`python
N_CTX = 16384        # unified-KV 全局池 (4 slot 共享)
N_PARALLEL = 4       # GPU 并发 slot
_GATE = Semaphore(4) # Python 层限流对齐 llama-server
max_tokens cap = 1024  # ctx 安全区
auto-split 阈值 = 2500 chars
`

### 性能迭代 (San Albino 16.1MB/33页)

| 版本 | 配置 | 耗时 | vs 基准 |
|------|------|------|---------|
| 基准 | rich开,RL=1,qps=4 | 341s | - |
| + rich关 | disable_rich_text=True | 313s | -8% |
| + RL放开 | RL=8,qps=16 | **254s** | **-26%** |

### 三层限流

`
RateLimiter(8) -> BabelDOC qps=16 -> _GATE(4) -> llama-server --parallel 4
  每125ms        21线程池            Python层         GPU真正并发
`

## 架构

`
浏览器 -> Flask(:5555) -> run_babeldoc.py(BabelDOC v2)
  -> babeldoc_translator.py -> llama_http_engine.py(_GATE Semaphore(4))
  -> HTTP POST -> llama-server(:8090, unified-KV, --parallel 4)
  -> Qwen3.5-4B-Q4_K_M GPU推理
  -> 返回JSON -> 组装 -> BabelDOC还原排版 -> mono.pdf + dual.pdf
`

## 项目结构

`
Geo-Mine-Translator/
+-- llama_http_engine.py    # 核心: llama-server HTTP + 巨长段自动切分 + ctx overflow 降级
+-- run_babeldoc.py         # 入口: BabelDOC v2 API + 三层限流
+-- babeldoc_translator.py  # BabelDOC -> engine 桥接
+-- web_app.py              # Flask 后端
+-- index.html              # Web UI
+-- start_translate.bat     # 一键启动
+-- geo_terms_final.json    # 术语库 14000+ 条
+-- llama-cpp/              # llama-server 二进制 (0.4.0-dev)
+-- models-gguf/            # GGUF 模型目录
`

## 常见问题

**Q: 启动报 No GGUF model found?**
A: 下载模型到 models-gguf/, 见模型下载章节.

**Q: 翻译速度 < 15 tok/s?**
A: 检查 --reasoning off (最关键!) 和 -ngl -1 (必须 GPU offload).

**Q: OOM 显存不足?**
A: 降 -c 8192, 降 --parallel 2, 或换 0.8B 模型.

**Q: PDF 排版乱了?**
A: 扫描件加 --skip-scanned-detection, 字体缺失检查 	ranslated/.

## License

代码 MIT License. 模型遵循各自原始 License.

## 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [BabelDOC](https://github.com/funstory-ai/BabelDOC)
- [Qwen](https://github.com/QwenLM/Qwen)
- [Hunyuan-MT](https://github.com/Tencent-Hunyuan/HunyuanWorld)
