from time import time
import logging


def timeit(fun):
    def wrapper(*args, **kwargs):
        start_time = time()
        output = fun(*args, **kwargs)
        logging.info(
            f"{fun.__name__} execution time: {time()-start_time:.2f}s")
