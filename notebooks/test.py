import torch
from transformers import FlaxAutoModelForCausalLM
from transformers import MistralConfig
from transformers import FlaxMistralForCausalLM
config = MistralConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
# model = FlaxAutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", config=config, from_pt=True)
model = FlaxMistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", from_pt=True)
