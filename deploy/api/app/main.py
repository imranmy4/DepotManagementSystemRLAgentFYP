"""HTTP API of the depot system: orders, crane moves and the yard view."""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.db.operations import OperationError
from app.routers import moves, orders, yard

app = FastAPI(title="Container depot API")
app.include_router(orders.router)
app.include_router(moves.router)
app.include_router(yard.router)

@app.exception_handler(OperationError)
def operation_error(request: Request, exc: OperationError):
    """A request that breaks an order or yard rule: 409 with the reason."""
    return JSONResponse(status_code=409, content={"detail": str(exc)})

@app.get("/health")
def health():
    """Liveness check."""
    return {"status":"ok"}