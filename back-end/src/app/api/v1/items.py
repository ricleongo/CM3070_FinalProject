from fastapi import APIRouter, Depends

from src.app.schemas.item import ItemResponse, ItemCreate

router = APIRouter()

@router.get("/", response_model=list[ItemResponse])
async def read_items():
    return [{"id": 1, "name": "Laptop", "price": 999.99}]

@router.post("/", response_model=ItemResponse)
async def create_item(item: ItemCreate):
    return {**item.model_dump(), "id": 101}

