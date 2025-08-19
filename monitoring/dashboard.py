from fastapi import APIRouter
from fastapi.responses import RedirectResponse

router = APIRouter()

@router.get("/dashboard")
async def dashboard() -> RedirectResponse:
    """Redirect to the main static dashboard."""
    return RedirectResponse("/")
