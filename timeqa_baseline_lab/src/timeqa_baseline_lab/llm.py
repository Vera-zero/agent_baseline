from __future__ import annotations

import os
import time
from typing import List, Optional

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from .config import ModelConfig


class BaseGenerator:
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError

    def batch_generate(self, prompts: List[str], system_prompt: Optional[str] = None, batch_size: int = 1) -> List[str]:
        if not prompts:
            return []
        bsz = max(1, int(batch_size))
        outputs: List[str] = []
        for i in range(0, len(prompts), bsz):
            for prompt in prompts[i : i + bsz]:
                outputs.append(self.generate(prompt, system_prompt=system_prompt))
        return outputs


class HFGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.device = self._resolve_device(device)
        dtype = self._resolve_dtype(torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "auto" else None,
        )
        if self.device in {"cpu", "cuda"}:
            self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_dtype(self, dtype: str):
        if dtype == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype, "auto")

    def _chat_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt

    def _generate_from_texts(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.temperature > 0
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = self.temperature

        gen = self.model.generate(**inputs, **kwargs)
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        outputs = []
        for i, in_len in enumerate(input_lens):
            out = self.tokenizer.decode(gen[i][int(in_len) :], skip_special_tokens=True)
            outputs.append(out.strip())
        return outputs

    @torch.no_grad()
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        text = self._chat_text(prompt, system_prompt)
        return self._generate_from_texts([text])[0]

    @torch.no_grad()
    def batch_generate(self, prompts: List[str], system_prompt: Optional[str] = None, batch_size: int = 1) -> List[str]:
        if not prompts:
            return []
        bsz = max(1, int(batch_size))
        outputs: List[str] = []
        for i in range(0, len(prompts), bsz):
            batch_prompts = prompts[i : i + bsz]
            texts = [self._chat_text(p, system_prompt) for p in batch_prompts]
            outputs.extend(self._generate_from_texts(texts))
        return outputs


class APIGenerator(BaseGenerator):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: int = 180,
    ):
        self.model = model
        self.base_url = base_url
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max(1, int(max_retries))
        self.timeout = max(1, int(timeout))

        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Missing API key: set env var {api_key_env}")

        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, prompt: str, system_prompt: Optional[str]) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        payload = self._payload(prompt, system_prompt)
        last_err = None

        for i in range(self.max_retries):
            try:
                resp = self.session.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError(f"No choices in response: {data}")
                content = choices[0].get("message", {}).get("content", "")
                return str(content).strip()
            except Exception as e:
                last_err = e
                if i < self.max_retries - 1:
                    time.sleep(1.5 * (i + 1))
                continue

        raise RuntimeError(f"API generation failed after {self.max_retries} retries: {last_err}")


class VLLMGenerator(BaseGenerator):
    """Generator for local VLLM service (e.g., Qwen3-32B)"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "qwen3-32b",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: int = 180,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max(1, int(max_retries))
        self.timeout = max(1, int(timeout))

        # Create OpenAI client for local VLLM service
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"  # Local VLLM doesn't require API key
        )

    def _build_messages(self, prompt: str, system_prompt: Optional[str]) -> List[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = self._build_messages(prompt, system_prompt)
        last_err = None

        for i in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    timeout=self.timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                if i < self.max_retries - 1:
                    time.sleep(1.5 * (i + 1))
                continue

        raise RuntimeError(f"VLLM generation failed after {self.max_retries} retries: {last_err}")


def build_generator(cfg: ModelConfig) -> BaseGenerator:
    provider = (cfg.provider or "local").lower()
    if provider in {"hf", "local"}:
        return HFGenerator(
            model_name=cfg.model_name,
            device=cfg.device,
            torch_dtype=cfg.torch_dtype,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
    if provider in {"api", "remote"}:
        return APIGenerator(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            max_retries=cfg.max_retries,
            timeout=cfg.timeout,
        )
    if provider == "vllm":
        return VLLMGenerator(
            base_url=cfg.base_url,
            model_name=cfg.model_name,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            max_retries=cfg.max_retries,
            timeout=cfg.timeout,
        )
    raise ValueError(f"Unsupported model.provider: {cfg.provider}. Use local/hf, remote/api, or vllm")
