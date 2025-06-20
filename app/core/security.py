import httpx
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional

from app.core.config import settings

security = HTTPBearer()

class JWTAuthService:
    """Service for validating JWT tokens against dsp_ai_jwt service"""
    
    @staticmethod
    async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
        """
        Validate the JWT token against the dsp_ai_jwt service.
        Returns the decoded token payload if valid.
        """
        token = credentials.credentials
        
        # Call the dsp_ai_jwt validation endpoint
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    settings.JWT_AUTH_URL,
                    json={"token": token},
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid authentication token",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                return response.json()
        except httpx.RequestError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable",
            )

    @staticmethod
    async def validate_api_key(api_key: str) -> Dict[str, Any]:
        """
        Validate an API key against the dsp_ai_jwt service.
        Returns the API key metadata if valid.
        """
        # Implement API key validation logic here
        # This would call the appropriate endpoint in the dsp_ai_jwt service
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.JWT_AUTH_URL}/api-key",
                    json={"api_key": api_key},
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                return response.json()
        except httpx.RequestError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication service unavailable",
            )

def get_current_user(token_data: Dict[str, Any] = Depends(JWTAuthService.validate_token)) -> Dict[str, Any]:
    """Get the current authenticated user from token data"""
    return token_data


async def get_optional_current_user(
    authorization: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Optional authentication - allows API access without authentication for development.
    
    This should only be used in development/test environments!
    
    Returns a mock user when no authentication is provided.
    """
    if not authorization:
        # Return a mock user for development purposes
        return {"sub": "test-user", "name": "Test User", "is_dev": True}
    
    # Try to validate the token if one is provided
    try:
        token = authorization.replace("Bearer ", "")
        # Call JWT validation service (simplified for this example)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.JWT_AUTH_URL,
                json={"token": token},
                timeout=5.0
            )
            
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
        
    # Fallback to mock user if token validation fails
    return {"sub": "test-user", "name": "Test User", "is_dev": True}
