# [Phi3-Personal-AL-V1](https://huggingface.co/abdullahalbaki/phi3-personal-al-v1)


This model is a fine-tuned version of the **Microsoft Phi-3-mini-4k-instruct** architecture. It has been specifically trained using **QLoRA (4-bit quantization)** to provide personalized information about [**Alamin Sarkar**](https://huggingface.co/AlaminSarkar01).

## 🚀 Model Details
- **Developed by:** [Abdullah Al Baki](https://huggingface.co/abdullahalbaki/)
- **Base Model:** [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Language:** English
- **Fine-tuning Method:** SFT (Supervised Fine-Tuning) with LoRA
- **Training Tool:** Hugging Face `transformers`, `peft`, and `bitsandbytes`

## 🧠 Training Information
The model was trained on a custom dataset containing personal information, professional expertise, and biographical data of Alamin Sarkar.

- **Epochs:** 50
- **Learning Rate:** 5e-4
- **Quantization:** 4-bit (NF4)
- **Target Modules:** `qkv_proj`
- **Training Hardware:** NVIDIA T4 GPU (Google Colab)

## 🎯 Usage
You can use this model directly with the `transformers` library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "abdullahalbaki/phi3-personal-al-v1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

prompt = "<|user|>\nWho is Alamin Sarkar?<|end|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
⚠️ Limitations and Bias
This model is highly specialized to answer questions about a specific individual (Alamin Sarkar) and his technical background. It may still inherit the general biases and limitations of the original Phi-3 base model. It is recommended to use it as a personal assistant or for demonstration purposes.

🛠 Technical Skills of Abdullah Al Baki
Languages: Python (Advanced)

Frameworks: FastAPI, Flask, SQLAlchemy, NextJS

Automation: Browser Automation (Selenium, Playwright), RPA, n8n

AI/ML: LangChain, RAG Pipelines, Agentic Workflows

DevOps: Docker, AWS, CI/CD
