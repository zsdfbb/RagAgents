# -*- coding: utf-8 -*-

import logging
import os
import subprocess
import sys
from abc import abstractmethod
from ast import Dict, List
from math import log

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from zhipuai import ZhipuAI


class LLMBase:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def instantiate(self, model_path: str):
        pass

    @abstractmethod
    def chat(self, text: str, max_tokens: int = 8000):
        pass

    @abstractmethod
    def dialog(self):
        pass


class LLM_chatglm3_6b_8k(LLMBase):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(LLM_chatglm3_6b_8k, "_initialized"):
            LLM_chatglm3_6b_8k._initialized = True
            self.tokenizer = None
            self.model = None
            self.instantiated = False

    def instantiate(self, model_path="/home/zsdfbb/ssd_2t/ai_model/chatglm3-6b"):
        """
        默认情况下，模型以 FP16 精度加载，运行上述代码需要大概 13GB 显存。如果你的 GPU 显存有限，可以尝试以量化方式加载模型，使用方法如下：
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, device="cuda"
        )
        self.model = model.eval()
        self.instantiated = True

    def chat(self, text="", max_tokens=8000):
        if self.instantiated != True:
            print("Please instantiate the model first.")
            return "", []

        response, history = self.model.chat(
            self.tokenizer,
            text,
            max_length=max_tokens,
            history=[],
            temperature=0.1,
        )
        # logging.warning(f"{response}")
        # logging.warning(f"{history}")
        return response, history

    def dialog(self):
        if self.instantiated != True:
            print("Please instantiate the model first.")
            return

        # get the input from user
        while True:
            user_input = input(
                "\nPlease enter your query [To end the task, type 'exit()']: \n"
            )
            if user_input == "exit()":
                break
            resp, _ = self.chat(user_input)
            print(f"\nResp:\n {resp}")


class LLM_chatglm3_6b_32k(LLMBase):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(LLM_chatglm3_6b_32k, "_initialized"):
            LLM_chatglm3_6b_32k._initialized = True
            self.tokenizer = None
            self.model = None
            self.instantiated = False

    def instantiate(self, model_path="/home/zsdfbb/ssd_2t/ai_model/chatglm3-6b-32k"):
        """
        默认情况下，模型以 FP16 精度加载，运行上述代码需要大概 13GB 显存。如果你的 GPU 显存有限，可以尝试以量化方式加载模型，使用方法如下：
        model = AutoModel.from_pretrained("THUDM/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, device="cuda"
        )
        # self.model = model.quantize(8)
        self.instantiated = True

    def chat(self, text="", max_tokens=32768):
        if self.instantiated != True:
            print("Please instantiate the model first.")
            return "", []

        response, history = self.model.chat(
            self.tokenizer,
            text,
            max_length=max_tokens,
            history=[],
            temperature=0.1,
        )
        # logging.warning(f"{response}")
        # logging.warning(f"{history}")
        return response, history

    def dialog(self):
        if self.instantiated != True:
            print("Please instantiate the model first.")
            return

        # get the input from user
        while True:
            user_input = input(
                "\nPlease enter your query [To end the task, type 'exit()']: \n"
            )
            if user_input == "exit()":
                break
            resp, _ = self.chat(user_input)
            print(f"\nResp:\n {resp}")


class LLM_mistral_7b(LLMBase):
    def __init__(self):
        self.instantiated = False
        self.device = "cuda"
        pass

    def instantiate(
        self, model_path="/home/zsdfbb/ssd_2t/ai_model/Mistral-7B-Instruct-v0.2/"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.to(self.device)
        self.instantiated = True
        """
        cmd = "nvidia-smi"
        res = subprocess.check_output(cmd, shell=True).decode("utf-8")
        print(res)
        """

    def chat(self, text: str, max_tokens: int = 8000):
        messages = [
            {"role": "user", "content": "Can you answer me some questions?"},
            {
                "role": "assistant",
                "content": "Of course, be willing to serve you. And I'll keep the output as short as possible.",
            },
            {"role": "user", "content": f"{text}"},
        ]

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(self.device)

        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=2,
            pad_token_id=2,
        )
        decoded = self.tokenizer.batch_decode(generated_ids)

        # 一些格式化处理
        resp = decoded[0]
        # 分割返回结果， mistral 使用 [/INST] 分割多次对话
        res = resp.split("[/INST]")[-1]
        # 删除末尾的 </s>
        res = res[:-4]
        return res, []

    def dialog(self):
        pass


from transformers.generation.utils import GenerationConfig


class LLM_baichuan13b_4bit(LLMBase):
    _instance = None
    # /home/zsdfbb/ssd_2t/ai_model/Baichuan2-13B-Chat-4bits

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(LLM_baichuan13b_4bit, "_initialized"):
            LLM_baichuan13b_4bit._initialized = True
            self.tokenizer = None
            self.model = None
            self.instantiated = False

    def instantiate(
        self, model_path="/home/zsdfbb/ssd_2t/ai_model/Baichuan2-13B-Chat-4bits"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision="v2.0",
            use_fast=False,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision="v2.0",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_path, revision="v2.0"
        )
        self.model.generation_config.temperature = 0.1
        self.instantiated = True

    def chat(self, text="", max_tokens=8000):
        if self.instantiated != True:
            print("Please instantiate the model first.")
            return "", []
        messages = []
        messages.append({"role": "user", "content": text})
        response = self.model.chat(self.tokenizer, messages)
        return response, messages

    def dialog(self):
        pass


class LLM_chatglm4_web(LLMBase):
    def __init__(self):
        self.instantiated = False
        self.device = "cuda"
        self.api_key = os.environ.get("ZHIPU_AI_API_KEY")

    def instantiate(self, model_path=""):
        self.client = ZhipuAI(api_key=self.api_key)  # 填写您自己的APIKey

    def chat(self, text: str, max_tokens: int = 8000):
        messages = [
            {"role": "user", "content": "Can you answer me some questions?"},
            {"role": "assistant", "content": "Of course, be willing to serve you."},
            {"role": "user", "content": f"{text}"},
        ]
        response = self.client.chat.completions.create(
            model="glm-4", messages=messages, temperature=0.1
        )
        res = str(response.choices[0].message)
        res = res.strip()
        res = res.replace("\\n", "\n")
        return res, []

    def dialog(self):
        pass


# =============================================
# Default model
# =============================================
global default_llm
default_llm = None


def llm_init(model_name: str = "llm_chatglm3_6b_32k"):
    global default_llm
    if model_name == "llm_chatglm3_6b_8k":
        default_llm = LLM_chatglm3_6b_8k()
        logging.info("llm_chatglm3_6b_8k is instantiated")
    if model_name == "llm_chatglm3_6b_32k":
        default_llm = LLM_chatglm3_6b_32k()
        logging.info(f"llm_chatglm3_6b_32k is instantiated, llm is {default_llm}")
    if model_name == "llm_mistral_7b":
        default_llm = LLM_mistral_7b()
        logging.info(f"llm_mistral_7b is instantiated, llm is {default_llm}")

    if model_name == "llm_baichuan13b_4bit":
        default_llm = LLM_baichuan13b_4bit()
        logging.info(f"llm_mistral_7b is instantiated, llm is {default_llm}")

    if model_name == "llm_chatglm4_web":
        default_llm = LLM_chatglm4_web()
        logging.info(f"llm_chatglm4_web is instantiated, llm is {default_llm}")

    default_llm.instantiate()
    logging.info(f"llm_init is done, default_llm is {default_llm}")


def llm_get_default():
    global default_llm
    return default_llm


# =============================================
# TEST
# =============================================


if __name__ == "__main__":
    # 判断传入的参数是否正确
    if len(sys.argv) < 2:
        print("Usage:")
        sys.exit(1)
    # 获取传入的参数
    option = sys.argv[1]

    logging.root.setLevel(logging.WARNING)

    if option == "LLM_chatglm3_6b_8k_test1":
        llm = LLM_chatglm3_6b_8k()
        llm.instantiate()
        # llm.dialog()

        with open("../test_data/test_llm_question1.txt", "r", encoding="utf-8") as f:
            query = f.read()
            resp, _ = llm.chat(query)
            print(query)
            print(resp)

        with open("../test_data/test_llm_question2.txt", "r", encoding="utf-8") as f:
            query = f.read()
            resp, _ = llm.chat(query)
            print(query)
            print(resp)
        exit()

    if option == "LLM_mistral_7b_test1":
        llm = LLM_mistral_7b()
        llm.instantiate()
        text = "hello, who are you?"
        resp, _ = llm.chat(text=text)
        print("question:" + text)
        print(resp)
        exit()

    if option != "":
        llm = LLM_mistral_7b()
        # llm = LLM_chatglm4_web()
        llm.instantiate()
        with open("../test_data/" + option + ".txt", "r", encoding="utf-8") as f:
            query = f.read()
            resp, _ = llm.chat(query)
            print(query)
            print(resp)
        exit()
