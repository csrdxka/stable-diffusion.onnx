from transformers import CLIPTokenizerFast

# 1. Load your existing tokenizer directory (which has merges.txt, vocab.json, etc.)
tokenizer_fast = CLIPTokenizerFast.from_pretrained("./tokenizer")

# 2. Save it again. This will produce a single tokenizer.json file in the output folder.
tokenizer_fast.save_pretrained("./tokenizer_fast")
