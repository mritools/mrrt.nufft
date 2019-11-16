from math import ceil
import warnings

try:
    import cupy

    default_device = cupy.cuda.device.Device()
    default_context = cupy.cuda.stream.get_current_stream()
except ImportError:
    default_device = None
    default_context = None

__all__ = ["get_1D_block_table_gridding"]


def get_1D_block_table_gridding(
    n_samples, dev, BLOCKSIZE_TABLE=512, kernel=None
):
    from cupy.cuda.runtime import CUDARuntimeError
    from cupy.cuda.driver import CUDADriverError

    try:
        if kernel is not None:
            # may be limited by the number of registers used by the kernel
            max_threads_per_block = kernel.max_threads_per_block
        else:
            # device limit
            max_threads_per_block = dev.attributes["MaxBlockDimX"]
    except (AttributeError, KeyError, CUDARuntimeError, CUDADriverError):
        warnings.warn("Unable to autodetect maxThreadsPerBlock, trying 512...")
        max_threads_per_block = 512

    threads = (
        min(min(n_samples, BLOCKSIZE_TABLE), max_threads_per_block),
        1,
        1,
    )
    blocks = (ceil(n_samples / BLOCKSIZE_TABLE), 1, 1)
    return threads, blocks
