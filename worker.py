"""Worker entrypoint: registers the Large 4 chat workflow and starts polling."""
import asyncio

import mistralai.workflows as workflows

from workflows.large4 import Large4Workflow


async def main() -> None:
    await workflows.run_worker([Large4Workflow])


if __name__ == "__main__":
    asyncio.run(main())
