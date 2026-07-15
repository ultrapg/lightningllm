# LightningLLM

LightningLLM is an ultra-fast, Vulkan-accelerated local Large Language Model (LLM) inference engine written in pure Rust. Designed for performance-critical and resource-constrained environments, it utilizes memory mapping to ensure models run efficiently with a minimal RAM footprint, while streaming contexts from SSDs on demand.

It supports 10 default performance tiers ranging from 135M to 4B parameters, featuring state-of-the-art models like SmolLM2, Qwen3, Gemma-3, Llama 3.2, and Phi-4-mini.

---

## Features

- **Vulkan GPU Acceleration & CPU Fallback**: Automatically offloads computation layers to the GPU via Vulkan for maximum inference speeds, falling back to CPU when necessary.
- **10 Logically Sorted Tiers**: Selectable tiers from `pico` (~135M parameters) to `xxl` (~4B parameters).
- **Dynamic Parameter Overrides**: Customize temperature, top_p, context size, and GPU layer offloading directly from the command line.
- **Output-Only Mode (`--output-only`)**: Silences all internal initialization, device logging, and progress statements, making it plug-and-play for scripts and integration.
- **Stateful Continuous Chat (`--chat <FILE.json>`)**: Automatically saves, parses, and appends conversation turns in an OpenAI-compatible JSON structure.
- **Reasoning Filtering**: Automatically filters out `<think>` / `<thinking>` reasoning sequences in the console output stream (hidden by default, enabled with `--show-thinking`).
- **Atomic Downloading**: Model downloads utilize temporary files (`.tmp`) and are only promoted upon success to prevent model file corruption.

---

## Selectable Performance Tiers

Use the `--list-tiers` command to see the available models:

- `pico` : SmolLM2-135M-Instruct-Q4_K_M (ChatML template)
- `nano` : SmolLM2-360M-Instruct-Q4_K_M (ChatML template)
- `micro` : Qwen3-0.6B-Instruct-Q4_K_M (Qwen/ChatML template)
- `tiny` : Gemma-3-1B-it-Q4_K_M (Gemma/Llama template)
- `mini` : Llama-3.2-1B-Instruct-Q4_K_M (Llama template)
- `small` : Qwen3-1.7B-Instruct-Q4_K_M (Qwen/ChatML template)
- `medium` : Llama-3.2-3B-Instruct-Q4_K_M (Llama template) [Default]
- `large` : Phi-4-mini-Instruct-Q4_K_M (Phi/Llama template)
- `xl` : Gemma-3-4B-it-Q4_K_M (Gemma/Llama template)
- `xxl` : Qwen3-4B-Instruct-Q4_K_M (Qwen/ChatML template)

---

## Build Requirements

1. **Rust Toolchain**: `cargo` & `rustc` (Edition 2021).
2. **Vulkan SDK**: Required for compiling shaders and linking the Vulkan backend libraries.
3. **Clang / Bindgen**: Required for compiling the `llama-cpp` bindings.

To build the project in optimized release mode:

```bash
# Set path to Vulkan SDK and Bindgen arguments if compiling manually
export VULKAN_SDK="/path/to/vulkansdk"
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:$LD_LIBRARY_PATH"
export BINDGEN_EXTRA_CLANG_ARGS="-I$VULKAN_SDK/include"

cargo build --release
```

---

## Usage Guide

The application reads configurations from `config.json` situated next to the binary. If it does not exist, it will be generated automatically.

### Commands

```bash
# 1. List all logically sorted performance tiers
./lightningllm --list-tiers

# 2. Start the interactive chat loop with the default model (medium)
./lightningllm

# 3. Start a chat loop with a specific model (e.g., small)
./lightningllm --tier small

# 4. Run a single prompt (outputting ONLY the model's response)
./lightningllm --tier pico -p "What is the capital of France?" --output-only

# 5. Continuous chat history using a JSON state file
./lightningllm --tier pico -p "What is 2+2?" --chat chat.json --output-only
./lightningllm --tier pico -p "Multiply that by 10." --chat chat.json --output-only

# 6. Override sampling parameters at execution time
./lightningllm --tier micro -p "Say OK." --temp 0.1 --top-p 0.95 --ctx 512 --output-only

# 7. Enable reasoning/thinking output for reasoning models
./lightningllm --tier micro -p "Why is the sky blue?" --show-thinking
```

---

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.
