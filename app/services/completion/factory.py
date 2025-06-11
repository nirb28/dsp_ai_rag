from typing import Dict, Type

from app.models.config_options import CompletionModelConfig, CompletionModelType
from app.services.completion.base import BaseCompletion
from app.services.completion.openai_completion import OpenAICompletion
from app.services.completion.local_completion import LocalCompletion


class CompletionFactory:
    """Factory for creating completion model instances"""
    
    # Registry of completion model classes
    _completion_models: Dict[CompletionModelType, Type[BaseCompletion]] = {
        CompletionModelType.OPENAI: OpenAICompletion,
        CompletionModelType.LOCAL: LocalCompletion,
    }
    
    @classmethod
    async def create_completion_model(cls, config: CompletionModelConfig) -> BaseCompletion:
        """
        Create a completion model instance based on the provided configuration.
        
        Args:
            config: Completion model configuration
            
        Returns:
            An instance of the appropriate completion model
            
        Raises:
            ValueError: If the requested completion model type is not supported
        """
        model_class = cls._completion_models.get(config.model_type)
        
        if not model_class:
            raise ValueError(f"Unsupported completion model type: {config.model_type}")
        
        model = model_class(config)
        await model.initialize()
        
        return model
