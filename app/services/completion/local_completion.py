from typing import List, Dict, Any, Optional, AsyncIterator, Tuple
import os
import asyncio
import torch
import logging

from app.models.config_options import CompletionModelConfig
from app.models.retrieval import RetrievedChunk, GenerationSource
from app.services.completion.base import BaseCompletion
from app.core.config import settings


class LocalCompletion(BaseCompletion):
    """Completion model using locally hosted models"""
    
    async def initialize(self):
        """Initialize the local completion model"""
        try:
            # Check if model path is provided
            if not self.config.local_model_path:
                raise ValueError("Local model path is required for local completion")
            
            # Construct full model path from LOCAL_MODELS_DIR and the provided path
            self.model_path = os.path.join(settings.LOCAL_MODELS_DIR, self.config.local_model_path)
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model path does not exist: {self.model_path}")
            
            # Import necessary libraries
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
                from transformers import pipeline
            except ImportError:
                raise ImportError("Please install transformers package with: pip install transformers")
            
            # Store the imports for later use
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
            self.TextIteratorStreamer = TextIteratorStreamer
            self.pipeline = pipeline
            
            # Set device
            self.device = self.config.device
            if self.device == "cuda" and not torch.cuda.is_available():
                logging.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Check for 8-bit or 4-bit quantization
            self.load_in_8bit = self.config.load_in_8bit
            self.load_in_4bit = self.config.load_in_4bit
            
            # Load tokenizer
            self.tokenizer = self.AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with appropriate quantization
            kwargs = {"device_map": self.device, "trust_remote_code": True}
            
            if self.load_in_8bit:
                kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                kwargs["load_in_4bit"] = True
                
            self.model = self.AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **kwargs
            )
            
            # Get max sequence length
            self.max_length = self.config.max_tokens or self.tokenizer.model_max_length
            if self.max_length > self.tokenizer.model_max_length:
                self.max_length = self.tokenizer.model_max_length
                
            # Create text generation pipeline for non-streaming
            self.text_generation = self.pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local completion model: {str(e)}")
    
    async def generate(
        self, 
        prompt: str, 
        context: Optional[List[RetrievedChunk]] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate a completion using a local model"""
        if not hasattr(self, "text_generation"):
            await self.initialize()
        
        # Format prompt with context if provided
        if context:
            formatted_prompt = self.format_prompt(prompt, context)
        else:
            formatted_prompt = prompt
        
        # Get generation parameters
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens or 512)
        temperature = kwargs.get("temperature", self.config.temperature or 0.7)
        top_p = kwargs.get("top_p", self.config.top_p or 0.95)
        repetition_penalty = kwargs.get("repetition_penalty", self.config.repetition_penalty or 1.1)
        
        try:
            # Run in a separate thread to not block the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.text_generation(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )
            )
            
            # Extract generated text
            generated_text = result[0]["generated_text"]
            
            # Remove prompt from the beginning
            if generated_text.startswith(formatted_prompt):
                completion_text = generated_text[len(formatted_prompt):]
            else:
                completion_text = generated_text
            
            # Prepare metadata
            generation_info = {
                "model": f"local:{self.config.local_model_path}",
                "source": GenerationSource.LOCAL,
            }
            
            return completion_text, generation_info
            
        except Exception as e:
            return f"Error generating completion with local model: {str(e)}", {"error": str(e)}
    
    async def generate_stream(
        self, 
        prompt: str, 
        context: Optional[List[RetrievedChunk]] = None,
        **kwargs
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Stream a completion using a local model"""
        if not hasattr(self, "model"):
            await self.initialize()
        
        # Format prompt with context if provided
        if context:
            formatted_prompt = self.format_prompt(prompt, context)
        else:
            formatted_prompt = prompt
        
        # Get generation parameters
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens or 512)
        temperature = kwargs.get("temperature", self.config.temperature or 0.7)
        top_p = kwargs.get("top_p", self.config.top_p or 0.95)
        repetition_penalty = kwargs.get("repetition_penalty", self.config.repetition_penalty or 1.1)
        
        try:
            # Prepare generation config
            streamer = self.TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                timeout=10.0
            )
            
            # Tokenize the prompt
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            # Start generation in a separate thread
            generation_kwargs = dict(
                inputs=inputs["input_ids"],
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
            
            # Start generation in a separate thread to not block the event loop
            loop = asyncio.get_event_loop()
            thread = loop.run_in_executor(None, lambda: self.model.generate(**generation_kwargs))
            
            # Prepare metadata
            generation_info = {
                "model": f"local:{self.config.local_model_path}",
                "source": GenerationSource.LOCAL,
            }
            
            # Stream the output
            for token in streamer:
                yield token, generation_info
                
            # Wait for the generation to complete
            await thread
            
        except Exception as e:
            yield f"Error generating stream with local model: {str(e)}", {"error": str(e)}
    
    def get_model_name(self) -> str:
        """Get the name of the completion model"""
        if hasattr(self, "model_path"):
            return f"local:{os.path.basename(self.model_path)}"
        return f"local:{self.config.local_model_path}"
    
    def format_prompt(self, prompt: str, context: List[RetrievedChunk]) -> str:
        """Format prompt with context for local models"""
        # Different models may require different formatting
        # Here's a template that works well with most instruction-tuned models
        formatted_prompt = "### System:\n"
        formatted_prompt += "You are a helpful assistant that answers questions based on the provided context.\n\n"
        
        formatted_prompt += "### Context:\n"
        
        # Add context sections
        for i, chunk in enumerate(context):
            # Add source information
            source_info = f"Source {i+1} [Document: {chunk.document_id}"
            if chunk.page_number:
                source_info += f", Page: {chunk.page_number}"
            source_info += "]"
            
            formatted_prompt += f"{source_info}\n{chunk.content}\n\n"
        
        # Add the query
        formatted_prompt += f"### User:\n{prompt}\n\n"
        formatted_prompt += "### Assistant:\n"
        
        return formatted_prompt
