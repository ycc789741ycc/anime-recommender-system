import asyncio
from typing import Any, Coroutine, List


class AioBatchConcurrency():
    """
    Seperating the coroutines into batches and run each batch coroutines individually.

    Example:
    >>> couroutines = [test1(), test2(), ...]
    >>> concurrency_limiter = AioBatchConcurrency(batch_size=10)
    >>> results = await concurrency_limiter.run_batch_concurrency(
        coroutines=couroutines
        )

    """
    def __init__(self, batch_size) -> None:
        self.semaphore = asyncio.Semaphore(batch_size)

    async def run_batch_concurrency(self, coroutines: List[Coroutine]) -> List[Any]:
        async def semaphore_wrapper(coroutine: Coroutine) -> Coroutine:
            async with self.semaphore:
                result = await coroutine
            return result

        futures = []
        results = []
        for coroutine in coroutines:
            futures.append(
                asyncio.create_task(semaphore_wrapper(coroutine))
            )
        for future in futures:
            results.append(await future)

        return results
