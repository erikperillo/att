import asyncio
import pickle
import gzip

async def ticker(delay, to):
    for i in range(to):
        print("loading...", flush=True)
        with gzip.open("/home/erik/data/saliency_datasets/salicon/train_pts/fixpts_1.gz") as f:
            wow = pickle.load(f)
        print("done.")
        yield i
        await asyncio.sleep(0.01)

async def run():
    import time
    async for i in ticker(1, 10):
        time.sleep(10)
        print(i)

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(run())
finally:
    loop.close()
