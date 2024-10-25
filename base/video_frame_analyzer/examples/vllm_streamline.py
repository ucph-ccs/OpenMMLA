import gc

import torch
from PIL import Image
from pupil_apriltags import Detector
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from openmmla.services.video.video_frame_analyzer import generate_vlm_prompt_msg, generate_llm_prompt_msg, \
    parse_text_to_dict
from openmmla.utils.video.apriltag import detect_apriltags

# Main execution
image_path = '../../llm-video-analyzer/data/llm_test/6.jpg'
detector = Detector(families='tag36h11', nthreads=4)
id_positions = detect_apriltags(image_path, detector, show=False)

# Vision Language Model
vlm_model_name = "openbmb/MiniCPM-V-2_6"
vlm_tokenizer = AutoTokenizer.from_pretrained(vlm_model_name, trust_remote_code=True)
vlm = LLM(model=vlm_model_name, trust_remote_code=True, gpu_memory_utilization=1, max_model_len=2048,
          enforce_eager=True)

vlm_messages = generate_vlm_prompt_msg(id_positions)
image = Image.open(image_path).convert("RGB")
vlm_prompt = vlm_tokenizer.apply_chat_template(vlm_messages, tokenize=False, add_generation_prompt=True)
vlm_inputs = {
    "prompt": vlm_prompt,
    "multi_modal_data": {"image": image}
}
vlm_stop_tokens = ['<|im_end|>', '<|endoftext|>']
vlm_stop_token_ids = [vlm_tokenizer.convert_tokens_to_ids(i) for i in vlm_stop_tokens]
vlm_sampling_params = SamplingParams(
    stop_token_ids=vlm_stop_token_ids,
    temperature=0,
    top_p=0.1,
    max_tokens=2048
)
vlm_outputs = vlm.generate(vlm_inputs, sampling_params=vlm_sampling_params)

for output in vlm_outputs:
    print("Output details:")
    print("  Finish reason:", output.outputs[0].finish_reason)
    print("  Prompt tokens:", len(output.prompt_token_ids))
    print("  Generated tokens:", len(output.outputs[0].token_ids))
    print("  Stop reason:", output.outputs[0].stop_reason)

image_description = vlm_outputs[0].outputs[0].text
print("Image Description:\n", image_description)

# Release resources
destroy_model_parallel()
del vlm.llm_engine.model_executor
del vlm
gc.collect()
torch.cuda.empty_cache()

# Language Model
llm_model_name = "microsoft/Phi-3-small-128k-instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
llm = LLM(model=llm_model_name, trust_remote_code=True, gpu_memory_utilization=0.8, max_model_len=2048,
          enforce_eager=True)

action_definitions = '''
'Reading': when a person is directly looking at and focusing on written or digital text, such as in a book, a printed document, or on a computer screen. This does not include times when they are using a writing instrument or mobile phone.\n
'Communicating': when a person speaking or listening, with their body and face oriented towards one or more individuals, displaying expressive gestures and facial interactions that signify active engagement.\n
'Writing': when a person is composing or editing text. This includes handwriting with a pen on paper or typing on a keyboard. This action is focused on text generation, not just passive reading or handling of papers.\n
'Unfocused': when a person is engaging with a mobile phone in ways that suggest leisure or distraction, like scrolling through social media, texting, or playing games.\n
'Unclear': you are not confident to categorize them into any of the above action classes.\n
'''
llm_messages = generate_llm_prompt_msg(image_description, action_definitions)
llm_prompt = llm_tokenizer.apply_chat_template(llm_messages, tokenize=False, add_generation_prompt=True)
llm_sampling_params = SamplingParams(temperature=0, top_p=0.1, max_tokens=1024)
llm_outputs = llm.generate(llm_prompt, llm_sampling_params)
categorization_result = llm_outputs[0].outputs[0].text
print("Categorization Result:", categorization_result)

# Release resources
destroy_model_parallel()
del llm.llm_engine.model_executor
del llm
gc.collect()
torch.cuda.empty_cache()

parsed_result = parse_text_to_dict(categorization_result)
print("Parsed Result:", parsed_result)
