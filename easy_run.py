# Script to easily run this model without actually doing much: 

!pip install -U transformers from transformers import pipeline

pipe = pipeline("text-generation", model="wildfellhall/public-poetry-50m")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("wildfellhall/public-poetry-50m") model = AutoModelForCausalLM.from_pretrained("wildfellhall/public-poetry-50m", device_map="auto")

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast import torch

saved_model = model.to("cuda" if torch.cuda.is_available() else "cpu")

title = input(" ") author = input(" ") prompt = ( f"<|startofpoem|>\n" f"<|title|> {title}\n" f"<|author|> {author}\n" f"<|body|> \n" )

inputs = tokenizer( prompt, return_tensors="pt").to(saved_model.device)

outputs = saved_model.generate( **inputs, max_new_tokens=150, temperature=0.8, top_p=0.9, repetition_penalty=1.15, do_sample=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
