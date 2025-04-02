from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_id = "microsoft/Phi-3-small-128k-instruct"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you? Please respond in pirate speak."},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Reduce GPU memory utilization
llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    max_model_len=1024,
    enforce_eager=True
)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
