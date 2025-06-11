from fastapi import APIRouter
from app.api.v1 import documents, retrievals, generations, configurations

# Create the main API router
api_router = APIRouter()

# Include sub-routers from each module
api_router.include_router(configurations.router, prefix="/config", tags=["Configuration"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(retrievals.router, prefix="/retrievals", tags=["Retrievals"])
api_router.include_router(generations.router, prefix="/generations", tags=["Generations"])
