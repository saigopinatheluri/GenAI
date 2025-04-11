import requests
import torch
import os
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer

# install nvidia-cuda-toolkit
# pip install flash_attn 

model_id = "ai4bharat/IndicTrans3-gemma-beta"
language = "Hindi"
# permitted languages = Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id)

src = [
    "When I was young, I used to go to the park every day.",
    "We watched a new movie last week, which was very inspiring.",
    "If you had met me at that time, we would have gone out to eat.",
    "My friend has invited me to his birthday party, and I will give him a gift."
]

_PROMPT = (
    "<bos><start_of_turn>user\n"
    "Translate the following text to {tgt_lang}: {source_text}:"
    "<end_of_turn>\n<start_of_turn>model\n"
)

batch_size = 100  # Adjust based on memory constraints
outputs = []

for i in tqdm(range(0, len(src), batch_size)):
    batch = src[i:i + batch_size]

    batch = [
        _PROMPT.format(
            tgt_lang=language,
            source_text=s
        )
        for s in batch
    ]
    tokinp = tokenizer(batch, return_tensors='pt', padding="longest")
    for k in tokinp:
        tokinp[k] = tokinp[k].to("cuda")

    out = model.generate(
        **tokinp,
        max_new_tokens=8192,
        num_beams=1,
        do_sample=False
    )

    for b, o in zip(batch, out):
        input_length = len(tokenizer(b)['input_ids'])
        finout = tokenizer.decode(o, skip_special_tokens=True)
        outputs.append(finout.split('model')[-1].strip())

print(outputs)
