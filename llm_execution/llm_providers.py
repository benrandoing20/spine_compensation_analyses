"""LLM provider implementations."""

import json
import logging
import time

import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger(__name__)


def parse_json_response(text):
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_openai(model_id, prompt, temperature, top_p, seed):
    """Query OpenAI API."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    start = time.time()
    
    # GPT-5 models have different parameter requirements
    is_gpt5 = model_id.startswith("gpt-5")
    
    # Build request parameters based on model type
    request_params = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    
    if is_gpt5:
        # GPT-5 uses max_completion_tokens and only supports default temperature (1)
        request_params["max_completion_tokens"] = 1000
        # Disable extended thinking/reasoning for faster responses
        request_params["reasoning_effort"] = "low"  # Options: low, medium, high
        request_params["store"] = False  # Don't store conversations
        # GPT-5 only supports temperature=1 (default), so we omit it
        # top_p and seed are also not supported or have restrictions
        logger.info(f"Using GPT-5 format for {model_id} with low reasoning effort for speed")
    else:
        # GPT-4 and earlier use max_tokens and support temperature/top_p/seed
        request_params["max_tokens"] = 1000
        request_params["temperature"] = temperature
        request_params["top_p"] = top_p
        request_params["seed"] = seed
    
    response = client.chat.completions.create(**request_params)
    
    text = response.choices[0].message.content
    
    return {
        "response": text,
        "parsed_response": parse_json_response(text),
        "metadata": {
            "elapsed_time": time.time() - start,
            "tokens": response.usage.total_tokens
        }
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_nvidia(model_id, prompt, temperature, top_p, seed):
    """Query NVIDIA API Catalog."""
    start = time.time()
    
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 2048,
            "seed": seed
        },
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    text = result["choices"][0]["message"]["content"]
    
    return {
        "response": text,
        "parsed_response": parse_json_response(text),
        "metadata": {
            "elapsed_time": time.time() - start,
            "tokens": result.get("usage", {}).get("total_tokens", 0)
        }
    }


def query_model(model_key, prompt, temperature, top_p, seed):
    """Query any model by key."""
    model_config = config.MODELS[model_key]
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    if provider == "openai":
        return query_openai(model_id, prompt, temperature, top_p, seed)
    elif provider == "nvidia":
        return query_nvidia(model_id, prompt, temperature, top_p, seed)
    else:
        raise ValueError(f"Unknown provider: {provider}")

