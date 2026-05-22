import asyncio
import json
import logging
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv()
# Imports below must run after load_dotenv() so modules that read env vars at
# import time (groq, sqlalchemy URL, etc.) see the .env values.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from groq import RateLimitError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from slowapi import Limiter  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi.middleware import SlowAPIMiddleware  # noqa: E402
from slowapi.util import get_remote_address  # noqa: E402
from sqlalchemy import text  # noqa: E402

logger = logging.getLogger("healthcare_rag")
logging.basicConfig(level=logging.INFO, format="%(message)s")

limiter = Limiter(key_func=get_remote_address)

QUERY_TIMEOUT_SECONDS = 90
# Groq's retry-after for TPM resets is typically <60s; TPD resets are hours.
# 5 minutes is a comfortable boundary: anything above it is a daily-scale wait.
RATE_LIMIT_LONG_THRESHOLD_SECONDS = 300


def _rate_limit_message(exc: RateLimitError) -> str:
    retry_after_raw = getattr(exc, "response", None) and exc.response.headers.get("retry-after")
    try:
        retry_after = float(retry_after_raw) if retry_after_raw else None
    except ValueError:
        retry_after = None

    if retry_after is None:
        return "The model is temporarily rate-limited. Please try again shortly."
    if retry_after >= RATE_LIMIT_LONG_THRESHOLD_SECONDS:
        hours = max(1, round(retry_after / 3600))
        unit = "hour" if hours == 1 else "hours"
        return (
            f"This demo has hit its daily quota with the model provider. "
            f"Please try again in about {hours} {unit}."
        )
    seconds = max(1, round(retry_after))
    unit = "second" if seconds == 1 else "seconds"
    return f"The model is temporarily rate-limited. Please try again in about {seconds} {unit}."


def _check_db() -> str:
    try:
        from healthcare_rag.core import get_engine
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        logger.exception("Database health check failed: %s", exc)
        return "error"


app = FastAPI(title="Healthcare RAG Agent")
app.state.limiter = limiter


@app.on_event("startup")
async def startup():
    from healthcare_rag.core import get_embed_model
    get_embed_model()


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={"type": "error", "message": "Too many requests. Please wait a moment and try again."},
    )

# Traffic can only come through my vercel or personal domain
ALLOWED_ORIGINS = (
    "https://healthcare-rag.alexdomnikov.com",
    "https://healthcare-rag-agent-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
)

app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()
    response = await call_next(request)
    logger.info(json.dumps({
        "request_id": request_id,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": int((time.time() - t0) * 1000),
    }))
    return response


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "qwen/qwen3-32b", "db": _check_db()}


@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, req: QueryRequest):
    async def event_stream():
        try:
            from healthcare_rag.core import get_agent
            agent = get_agent()
            tool_used: str | None = None

            async with asyncio.timeout(QUERY_TIMEOUT_SECONDS):
                async for event in agent.astream_events(
                    {"messages": [{"role": "user", "content": req.question}]},
                    version="v2",
                ):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'value': content})}\n\n"
                    elif kind == "on_tool_start":
                        tool_used = event["name"]
                        yield f"data: {json.dumps({'type': 'tool', 'name': tool_used})}\n\n"
                    elif kind == "on_chain_end" and event["name"] == "LangGraph":
                        yield f"data: {json.dumps({'type': 'done', 'tool_used': tool_used})}\n\n"

        except asyncio.TimeoutError:
            logger.warning("Query exceeded %ss timeout", QUERY_TIMEOUT_SECONDS)
            yield f"data: {json.dumps({'type': 'error', 'message': 'The request took too long and was cancelled. Please try a simpler question or try again.'})}\n\n"
        except RateLimitError as exc:
            retry_after = getattr(exc, "response", None) and exc.response.headers.get("retry-after")
            logger.warning("Groq rate limit hit on /query (retry-after=%s)", retry_after)
            yield f"data: {json.dumps({'type': 'error', 'message': _rate_limit_message(exc)})}\n\n"
        except Exception as exc:
            logger.exception("Unhandled error in event stream: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': 'An internal error occurred.'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
