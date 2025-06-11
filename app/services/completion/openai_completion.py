from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
import os
import httpx
import json
import asyncio

from app.models.config_options import CompletionModelConfig
from app.models.retrieval import RetrievedChunk, GenerationSource
from app.services.completion.base import BaseCompletion
from app.core.config import settings


class OpenAICompletion(BaseCompletion):
    """Completion model using OpenAI's API"""
    
    async def initialize(self):
        """Initialize the OpenAI completion model"""
        # Set up API key
        self.api_key = self.config.openai_api_key or settings.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required for OpenAI completion")
        
        # Set default model if not specified
        self.model_name = self.config.model_name or "gpt-3.5-turbo"
        
        # Base URL (allowing for custom endpoints like Azure)
        self.api_base = self.config.api_base_url or "https://api.openai.com/v1"
        self.api_type = self.config.api_type or "openai"
        self.api_version = self.config.api_version
    
    async def generate(
        self, 
        prompt: str, 
        context: Optional[List[RetrievedChunk]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a completion using OpenAI's API"""
        if not hasattr(self, "api_key"):
            await self.initialize()
        
        # Prepare prompt with context if provided
        if context:
            system_prompt = self.config.system_prompt or "You are a helpful assistant that provides accurate information based on the given context."
            formatted_prompt = self.format_prompt(prompt, context)
        else:
            system_prompt = self.config.system_prompt or "You are a helpful assistant."
            formatted_prompt = prompt
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature or 0.7),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 1000),
            "top_p": kwargs.get("top_p", self.config.top_p or 1),
        }
        
        # Set frequency and presence penalties if provided
        if "frequency_penalty" in kwargs or self.config.frequency_penalty is not None:
            payload["frequency_penalty"] = kwargs.get("frequency_penalty", self.config.frequency_penalty)
        
        if "presence_penalty" in kwargs or self.config.presence_penalty is not None:
            payload["presence_penalty"] = kwargs.get("presence_penalty", self.config.presence_penalty)
        
        # Set the appropriate API URL
        if self.api_type == "azure":
            api_url = f"{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
        else:  # openai
            api_url = f"{self.api_base}/chat/completions"
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_type == "azure":
            headers["api-key"] = self.api_key
        else:  # openai
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"OpenAI API error: {response.status_code} - {response.text}")
                
                response_data = response.json()
                
                # Extract the completion text
                completion_text = response_data["choices"][0]["message"]["content"]
                
                # Extract metadata
                generation_info = {
                    "model": self.model_name,
                    "finish_reason": response_data["choices"][0].get("finish_reason"),
                    "source": GenerationSource.OPENAI,
                }
                
                # Add usage statistics if available
                if "usage" in response_data:
                    generation_info["usage"] = response_data["usage"]
                
                return completion_text, generation_info
                
        except Exception as e:
            return f"Error generating completion: {str(e)}", {"error": str(e)}
    
    async def generate_stream(
        self, 
        prompt: str, 
        context: Optional[List[RetrievedChunk]] = None,
        **kwargs
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Stream a completion using OpenAI's API"""
        if not hasattr(self, "api_key"):
            await self.initialize()
        
        # Prepare prompt with context if provided
        if context:
            system_prompt = self.config.system_prompt or "You are a helpful assistant that provides accurate information based on the given context."
            formatted_prompt = self.format_prompt(prompt, context)
        else:
            system_prompt = self.config.system_prompt or "You are a helpful assistant."
            formatted_prompt = prompt
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature or 0.7),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 1000),
            "top_p": kwargs.get("top_p", self.config.top_p or 1),
            "stream": True  # Enable streaming
        }
        
        # Set frequency and presence penalties if provided
        if "frequency_penalty" in kwargs or self.config.frequency_penalty is not None:
            payload["frequency_penalty"] = kwargs.get("frequency_penalty", self.config.frequency_penalty)
        
        if "presence_penalty" in kwargs or self.config.presence_penalty is not None:
            payload["presence_penalty"] = kwargs.get("presence_penalty", self.config.presence_penalty)
        
        # Set the appropriate API URL
        if self.api_type == "azure":
            api_url = f"{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
        else:  # openai
            api_url = f"{self.api_base}/chat/completions"
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_type == "azure":
            headers["api-key"] = self.api_key
        else:  # openai
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=60.0
                ) as response:
                    if response.status_code != 200:
                        error_detail = await response.text()
                        raise RuntimeError(f"OpenAI API error: {response.status_code} - {error_detail}")
                    
                    # Initialize metadata
                    generation_info = {
                        "model": self.model_name,
                        "source": GenerationSource.OPENAI,
                    }
                    
                    # Process the stream
                    async for line in response.aiter_lines():
                        if line.strip() and line.strip() != "data: [DONE]":
                            # Extract the JSON data from the SSE format
                            if line.startswith("data: "):
                                json_str = line[6:]  # Remove "data: " prefix
                                try:
                                    data = json.loads(json_str)
                                    
                                    # Extract delta content if available
                                    if (
                                        "choices" in data and 
                                        len(data["choices"]) > 0 and 
                                        "delta" in data["choices"][0] and 
                                        "content" in data["choices"][0]["delta"]
                                    ):
                                        delta_content = data["choices"][0]["delta"]["content"]
                                        
                                        # Update finish reason if available
                                        if "finish_reason" in data["choices"][0] and data["choices"][0]["finish_reason"]:
                                            generation_info["finish_reason"] = data["choices"][0]["finish_reason"]
                                        
                                        yield delta_content, generation_info
                                    
                                except json.JSONDecodeError:
                                    continue
                
        except Exception as e:
            yield f"Error generating stream: {str(e)}", {"error": str(e)}
    
    def get_model_name(self) -> str:
        """Get the name of the completion model"""
        if hasattr(self, "model_name"):
            return f"openai:{self.model_name}"
        return "openai:gpt-3.5-turbo"
    
    def format_prompt(self, prompt: str, context: List[RetrievedChunk]) -> str:
        """Format prompt with retrieved context"""
        formatted_prompt = "I need an answer based on the following information:\n\n"
        
        # Add context sections with source information
        for i, chunk in enumerate(context):
            # Add source information
            source_info = f"Source {i+1} [Document: {chunk.document_id}"
            if chunk.page_number:
                source_info += f", Page: {chunk.page_number}"
            source_info += "]"
            
            formatted_prompt += f"{source_info}\n{chunk.content}\n\n"
        
        # Add the query
        formatted_prompt += f"Question: {prompt}\n\n"
        formatted_prompt += "Provide a detailed answer using only the information provided in the sources above. If the information needed is not in the sources, say 'I don't have enough information to answer this question.'"
        
        return formatted_prompt
