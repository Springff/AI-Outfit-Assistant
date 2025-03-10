import os
from modelscope import snapshot_download
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import  AutoTokenizer

def install_model():
    model_dir = snapshot_download('Qwen/Qwen2-7B-Instruct', cache_dir='qwen2chat_src', revision='master')
    model_path = os.path.join(os.getcwd(),"qwen2chat_src/Qwen/Qwen2-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int8', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.save_low_bit('qwen2chat_int8')
    tokenizer.save_pretrained('qwen2chat_int8')
    # model_dir = snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir='qwen2chat_src', revision='master') 

if __name__ == "__main__":
    install_model()
    print("7B模型下载完成")
    