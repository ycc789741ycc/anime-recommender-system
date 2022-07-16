import asyncio
import functools
import logging
from typing import Coroutine
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


def execute_in_running_event_loop_until_task_done(coroutine: Coroutine):
    """
    There's no way to directly await the coroutine in the running eventloop,
    this comes to a big problem while using jupyter.
    This function implement the way to run coroutine function in running
    eventloop and retrieve the result when coroutine is done but eventloop
    still running.
    """
    result_stack = [None]

    async def coro_wrapper(coroutine: Coroutine):
        result_stack.append(await coroutine)

    event_loop = asyncio.get_event_loop()
    if event_loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            exec_func = functools.partial(asyncio.run, coro_wrapper(coroutine))
            event_loop.run_in_executor(executor=executor, func=exec_func)

        return result_stack.pop()
    else:
        logger.info("There's no running event loop.")
        done, pending = event_loop.run_until_complete(
            asyncio.wait(coroutine)
        )
        return done
