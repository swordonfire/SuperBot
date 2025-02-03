from fastapi import FastAPI

from src.core.config.settings import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.PROJECT_NAME)

    @app.get('/')
    async def root():
        return {'message': f'Welcome to {settings.PROJECT_NAME} API'}

    return app


app = create_app()

# Import routes AFTER app creation to avoid circular imports
from src.api.routes.llm import router as llm_router  # noqa: E402

app.include_router(llm_router, prefix='/api/v1')
