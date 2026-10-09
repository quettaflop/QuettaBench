"""CPU stand-in for the two torch.cuda calls vmin_fit.py makes."""

GPU_NAME = "STUB-GPU-48GiB"
GPU_BYTES = 48 * (1 << 30)


class _Props:
    total_memory = GPU_BYTES


class cuda:
    @staticmethod
    def get_device_name(index):
        return GPU_NAME

    @staticmethod
    def get_device_properties(index):
        return _Props()
