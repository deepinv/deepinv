from __future__ import annotations
import os
import csv
from datetime import datetime
import platform
import numpy as np
from deepinv.utils.decorators import _deprecated_class, _deprecated_func


class AverageMeter:
    """Compute and store aggregates online from a stream of scalar values

    The supported aggregates are:
    - vals: the list of all processed values
    - val: the last value processed
    - avg: the average of all processed values
    - sum: the sum of all processed values
    - count: the number of processed values
    - std: the standard deviation of all processed values
    - sum2: the sum of squares of all processed values

    :param str name: meter name for printing
    :param str fmt: meter format for printing
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        """Reset the stored aggregates."""
        self.vals = []
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.std = 0.0
        self.sum2 = 0.0

    def update(self, val: np.ndarray | float | int, n: int = 1) -> None:
        """Process new scalar value(s) and update the stored aggregates.

        :param numpy.ndarray, float, int val: either array (i.e. batch) of values or single value
        :param int n: weight, defaults to 1
        """
        if isinstance(val, np.ndarray):
            # NOTE: numpy.ndarray.tolist converts values to native Python types
            self.vals += val.tolist() if val.ndim > 0 else [val.tolist()]
            self.val = float(np.mean(val))
            self.sum += float(np.sum(val) * n)
            self.sum2 += float(np.sum(val**2) * n)
            self.count += float(n * np.prod(val.shape))
        else:
            self.vals += [float(val)]
            self.val = float(val)
            self.sum += float(val * n)
            self.sum2 += float(val**2 * n)
            self.count += float(n)

        self.avg = float(self.sum / self.count)
        var = self.sum2 / self.count - self.avg**2
        self.std = float(np.sqrt(var) if var > 0 else 0)

    def __str__(self):
        fmtstr = "{name}={avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


@_deprecated_class
class ProgressMeter:
    def __init__(self, num_epochs, meters, surfix="", prefix=""):
        self.epoch_fmtstr = self._get_epoch_fmtstr(num_epochs)
        self.meters = meters
        self.surfix = surfix
        self.prefix = prefix

    def display(self, epoch):
        entries = [self.surfix]
        entries += [get_timestamp()]
        entries += [self.epoch_fmtstr.format(epoch)]
        entries += [str(meter) for meter in self.meters]
        entries += [self.prefix]
        print("\t".join(entries))

    def _get_epoch_fmtstr(self, num_epochs):
        num_digits = len(str(num_epochs // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_epochs) + "]"


def get_timestamp() -> str:
    """Get current timestamp string.

    :return str: timestamp, with separators determined by system.
    """
    sep = "_" if platform.system() == "Windows" else ":"
    return datetime.now().strftime(f"%y-%m-%d-%H{sep}%M{sep}%S")


@_deprecated_class
class LOG:
    def __init__(self, filepath, filename, field_name):
        self.filepath = filepath
        self.filename = filename
        self.field_name = field_name

        self.logfile, self.logwriter = csv_log(
            file_name=os.path.join(filepath, filename + ".csv"), field_name=field_name
        )
        self.logwriter.writeheader()

    def record(self, *args):
        dict = {}
        for i in range(len(self.field_name)):
            dict[self.field_name[i]] = args[i]
        self.logwriter.writerow(dict)

    def close(self):
        self.logfile.close()

    def print(self, msg):
        logT(msg)


@_deprecated_func
def csv_log(file_name, field_name):
    assert file_name is not None
    assert field_name is not None
    logfile = open(file_name, "w")
    logwriter = csv.DictWriter(logfile, fieldnames=field_name)
    return logfile, logwriter


@_deprecated_func
def logT(*args, **kwargs):
    print(get_timestamp(), *args, **kwargs)
