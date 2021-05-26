import logging
import os
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

def setup_logger(name) -> logging.Logger:
    FORMAT = "[%(name)s %(module)s:%(lineno)s]\n\t %(message)s \n"
    TIME_FORMAT = "%d.%m.%Y %I:%M:%S %p"
    print 
    logging.basicConfig(
        format=FORMAT, datefmt=TIME_FORMAT, level=logging.INFO, filename="req_res.log"
    )

    logger = logging.getLogger(name)
    return logger

async def set_body(request: Request, body: bytes):
    async def receive() -> Message:
        return {"type": "http.request", "body": body}
    request._receive = receive
 
async def get_body(request: Request) -> bytes:
    body = await request.body()
    await set_body(request, body)
    return body