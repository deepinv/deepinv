import torch
import os
import numpy as np


class TensorList:
    r"""

    Represents a list of :class:`torch.Tensor` with different shapes.
    It allows to sum, flatten, append, etc. lists of tensors seamlessly, in a
    similar fashion to :class:`torch.Tensor`.

    :param x: a list of :class:`torch.Tensor`, a single :class:`torch.Tensor` or a TensorList.
    """

    def __init__(self, x):
        super().__init__()

        if isinstance(x, list) or isinstance(x, TensorList):
            self.x = list(x)
        elif isinstance(x, torch.Tensor):
            self.x = [x]
        else:
            raise TypeError("x must be a list of torch.Tensor or a single torch.Tensor")

        self.shape = [xi.shape for xi in self.x]

    def __repr__(self):
        return f"TensorList({self.x})"

    def __len__(self):
        r"""
        Returns the number of tensors in the list.
        """
        return len(self.x)

    def __getitem__(self, item):
        r"""
        Returns the ith tensor in the list.
        """
        return self.x[item]

    def flatten(self):
        r"""
        Returns a :class:`torch.Tensor` with a flattened version of the list of tensors.
        """
        return torch.cat([xi.flatten() for xi in self.x])

    def append(self, other):
        r"""
        Appends a :class:`torch.Tensor` or a list of :class:`torch.Tensor` to the list.

        """
        if isinstance(other, list):
            self.x += other
        elif isinstance(other, TensorList):
            self.x += other.x
        elif isinstance(other, torch.Tensor):
            self.x.append(other)
        else:
            raise TypeError(
                "the appended item must be a list of :class:`torch.Tensor` or a single :class:`torch.Tensor`"
            )
        return self

    def __add__(self, other):
        r"""

        Adds two TensorLists. The sizes of the tensor lists must match.

        """
        if not isinstance(other, list) and not isinstance(other, TensorList):
            return TensorList([xi + other for xi in self.x])
        else:
            return TensorList([xi + otheri for xi, otheri in zip(self.x, other)])

    def __mul__(self, other):
        r"""

        Multiply two TensorLists. The sizes of the tensor lists must match.

        """
        if not isinstance(other, list) and not isinstance(other, TensorList):
            return TensorList([xi * other for xi in self.x])
        else:
            return TensorList([xi * otheri for xi, otheri in zip(self.x, other)])

    def __rmul__(self, other):
        r"""

        Multiply two TensorLists. The sizes of the tensor lists must match.

        """
        if not isinstance(other, list) and not isinstance(other, TensorList):
            return TensorList([xi * other for xi in self.x])
        else:
            return TensorList([xi * otheri for xi, otheri in zip(self.x, other)])

    def __truediv__(self, other):
        r"""

        Divide two TensorLists. The sizes of the tensor lists must match.

        """
        if not isinstance(other, list) and not isinstance(other, TensorList):
            return TensorList([xi / other for xi in self.x])
        else:
            return TensorList([xi / otheri for xi, otheri in zip(self.x, other)])

    def __neg__(self):
        r"""

        Negate a TensorList.
        """
        return TensorList([-xi for xi in self.x])

    def __sub__(self, other):
        r"""

        Substract two TensorLists. The sizes of the tensor lists must match.

        """
        if not isinstance(other, list) and not isinstance(other, TensorList):
            return TensorList([xi - other for xi in self.x])
        else:
            return TensorList([xi - otheri for xi, otheri in zip(self.x, other)])

    def conj(self):
        r"""

        Computes the conjugate of the elements of the TensorList.

        """
        return TensorList([xi.conj() for xi in self.x])

    def sum(self, dim, keepdim=False):
        r"""

        Computes the sum of each elements of the TensorList along the given dimension(s).

        """
        return TensorList([xi.sum(dim, keepdim) for xi in self.x])

    def reshape(self, shape):
        r"""

        Reshape each tensor of the TensorList into the given list of shapes.

        """
        return TensorList([self.x[i].reshape(shape[i]) for i in range(len(self.x))])

    def __any__(self):
        r"""

        Returns True if any of the elements of the TensorList is True.

        """
        return any([xi.any() for xi in self.x])

    def __all__(self):
        r"""

        Returns True if all the elements of the TensorList are True.

        """
        return all([xi.all() for xi in self.x])

    def __gt__(self, other):
        r"""

        Returns a TensorList of True if the elements of the input TensorList are greater than other.

        """

        return TensorList([xi > other for xi in self.x])

    def __lt__(self, other):
        r"""

        Returns a TensorList of True if the elements of the TensorList are smaller than other.

        """

        return TensorList([xi < other for xi in self.x])


def randn_like(x):
    r"""
    Returns a :class:`deepinv.utils.TensorList` or :class:`torch.Tensor`
    with the same type as x, filled with standard gaussian numbers.
    """
    if isinstance(x, torch.Tensor):
        return torch.randn_like(x)
    else:
        return TensorList([torch.randn_like(xi) for xi in x])


def rand_like(x):
    r"""
    Returns a :class:`deepinv.utils.TensorList` or :class:`torch.Tensor`
    with the same type as x, filled with random uniform numbers in [0,1].
    """
    if isinstance(x, torch.Tensor):
        return torch.rand_like(x)
    else:
        return TensorList([torch.rand_like(xi) for xi in x])


def zeros_like(x):
    r"""
    Returns a :class:`deepinv.utils.TensorList` or :class:`torch.Tensor`
    with the same type as x, filled with zeros.
    """
    if isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    else:
        return TensorList([torch.zeros_like(xi) for xi in x])


def ones_like(x):
    r"""
    Returns a :class:`deepinv.utils.TensorList` or :class:`torch.Tensor`
    with the same type as x, filled with ones.
    """
    if isinstance(x, torch.Tensor):
        return torch.ones_like(x)
    else:
        return TensorList([torch.ones_like(xi) for xi in x])


def get_freer_gpu(verbose=True):
    """
    Returns the GPU device with the most free memory.

    """
    try:
        if os.name == "posix":
            os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp")
            memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        else:
            os.system('bash -c "nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >tmp"')
            memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
        idx = np.argmax(memory_available)
        device = torch.device(f"cuda:{idx}")
        if verbose:
            print(f"Selected GPU {idx} with {np.max(memory_available)} MB free memory ")
    except:
        device = torch.device(f"cuda")
        print("Couldn't find free GPU")

    return device


def load_checkpoint(model, path_checkpoint, device):
    checkpoint = torch.load(path_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def investigate_model(model, idx_max=1, check_name="iterator.g_step.g_param.0"):
    for idx, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad and (idx < idx_max or check_name in name):
            print(
                name,
                param.data.flatten()[0],
                "gradient norm = ",
                param.grad.detach().data.norm(2),
            )
