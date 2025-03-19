from openai import OpenAI
import requests
import json

class DeepSeekNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instruction_title": ("STRING", {
                    "multiline": False,
                    "default": "↓↓↓ 输入指令 ↓↓↓"
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "Optimize the AI instruction..."
                }),
                "prompt_topic_title": ("STRING", {
                    "multiline": False,
                    "default": "↓↓↓ 输入主题 ↓↓↓"
                }),
                "prompt_topic": ("STRING", {
                    "multiline": True,
                    "default": "一个女人在沙滩上"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": ""
                }),
                "api_choice": (["deepseek", "siliconflow"],),
                "model": ("STRING", {
                    "multiline": False,
                    "default": "deepseek-reasoner"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "top_k": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("reasoning", "response")
    FUNCTION = "process"
    CATEGORY = "VisioStar"

    def process(self, instruction_title, instruction, prompt_topic_title, prompt_topic, api_key, api_choice, model, temperature, max_tokens, top_p, top_k=50, frequency_penalty=0.5):
        combined_text = f"{instruction}\n提示词主题：{prompt_topic}"

        try:
            if api_choice == "deepseek":
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                messages = [{"role": "user", "content": combined_text}]
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )

                reasoning_content = getattr(response.choices[0].message, 'reasoning_content', "No reasoning provided")
                content = response.choices[0].message.content

            elif api_choice == "siliconflow":
                url = "https://api.siliconflow.cn/v1/chat/completions"
                if model == "deepseek-reasoner":
                    model = "Qwen/QwQ-32B"
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": combined_text}],
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "frequency_penalty": frequency_penalty,
                    "n": 1,
                    "response_format": {"type": "text"}
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                response = requests.post(url, json=payload, headers=headers)

                if response.status_code == 200:
                    response_data = response.json()
                    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    reasoning_content = "No reasoning provided by SiliconFlow API"
                else:
                    error_message = f"API Error: {response.status_code} - {response.text}"
                    return (error_message, error_message)
            else:
                raise ValueError("Invalid API choice")

            return (reasoning_content, content)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            return (error_message, error_message)

NODE_CLASS_MAPPINGS = {
    "DeepSeekNode": DeepSeekNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekNode": "DeepSeek r1"
}
