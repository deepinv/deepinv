from typing import Union
import os
import csv
from datetime import datetime
import platform
import numpy as np


class AverageMeter(object):
    """Stores values and keeps track of averages and standard deviations.

    :param str name: meter name for printing
    :param str fmt: meter format for printing
    """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset meter values."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.std = 0.0
        self.sum2 = 0.0

    def update(self, val: Union[np.ndarray, float, int], n: int = 1) -> None:
        """Update average meter.

        :param numpy.ndarray, float, int val: either array (i.e. batch) of values or single value
        :param int n: weight, defaults to 1
        """
        if isinstance(val, np.ndarray):
            self.val = np.mean(val)
            self.sum += np.sum(val) * n
            self.sum2 += np.sum(val**2) * n
            self.count += n * np.prod(val.shape)
        else:
            self.val = val
            self.sum += val * n
            self.sum2 += val**2 * n
            self.count += n

        self.avg = self.sum / self.count
        var = self.sum2 / self.count - self.avg**2
        self.std = np.sqrt(var) if var > 0 else 0

    def __str__(self):
        fmtstr = "{name}={avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
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


class LOG(object):
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


def csv_log(file_name, field_name):
    assert file_name is not None
    assert field_name is not None
    logfile = open(file_name, "w")
    logwriter = csv.DictWriter(logfile, fieldnames=field_name)
    return logfile, logwriter


def logT(*args, **kwargs):
    print(get_timestamp(), *args, **kwargs)
