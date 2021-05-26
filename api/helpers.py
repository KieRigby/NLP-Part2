import logging
from loguru import logger
import os
import sys
from starlette.types import Message
from fastapi import Request

class async_iterator_wrapper:
    def __init__(self, obj):
        self._it = iter(obj)
    def __aiter__(self):
        return self
    async def __anext__(self):
        try:
            value = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        return value

LOG_LEVEL = logging.getLevelName(os.environ.get("LOG_LEVEL", "DEBUG"))
JSON_LOGS = True if os.environ.get("JSON_LOGS", "0") == "1" else False

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    # intercept everything at the root logger
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(LOG_LEVEL)

    # remove every other logger's handlers
    # and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    # configure loguru
    logger.configure(handlers=[{"sink": sys.stdout, "serialize": JSON_LOGS},{"sink":"req_res.log", "enqueue":True, "serialize":True}])

async def set_body(request: Request, body: bytes):
    async def receive() -> Message:
        return {"type": "http.request", "body": body}
    request._receive = receive
 
async def get_body(request: Request) -> bytes:
    body = await request.body()
    await set_body(request, body)
    return body