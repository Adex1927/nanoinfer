# Usage

```bash
#!/bin/bash

# Evaluate an empty prompt (just BOS token <s>):
python3 tools/dump_logits.py -m ../llama.cpp/models/tinyllama.gguf -p "" --top 10 --check 450 2760

# Evaluate any prompt (e.g., "Hello"):
python3 tools/dump_logits.py -m ../llama.cpp/models/tinyllama.gguf -p "Hello" --top 10

# Inspect a pre-existing .bin logits dump without running llama-debug:
python3 tools/dump_logits.py -m ../llama.cpp/models/tinyllama.gguf --bin /tmp/test_logits/llamacpp-tinyllama.bin --top 10
```
