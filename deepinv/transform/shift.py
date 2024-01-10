import torch


class Shift(torch.nn.Module):
    r"""
    Fast integer 2D translations.

    Generates n_transf randomly shifted versions of 2D images with circular padding.

    :param n_trans: number of shifted versions generated per input image.
    :param float shift_max: maximum shift as fraction of total height/width.
    """

    def __init__(self, n_trans=1, shift_max=1.0):
        super(Shift, self).__init__()
        self.n_trans = n_trans
        self.shift_max = shift_max

    def forward(self, x):
        r"""
        Applies a random translation to the input image.

        :param torch.Tensor x: input image
        :return: torch.Tensor containing the translated images concatenated along the first dimension
        """
        H, W = x.shape[-2:]
        assert self.n_trans <= H - 1 and self.n_trans <= W - 1

        H_max, W_max = int(self.shift_max * H), int(self.shift_max * W)

        x_shift = (
            torch.arange(-H_max, H_max)[torch.randperm(2 * H_max)][: self.n_trans]
            if H_max > 0
            else torch.zeros(self.n_trans)
        )
        y_shift = (
            torch.arange(-W_max, W_max)[torch.randperm(2 * W_max)][: self.n_trans]
            if W_max > 0
            else torch.zeros(self.n_trans)
        )

        out = torch.cat(
            [torch.roll(x, [sx, sy], [-2, -1]) for sx, sy in zip(x_shift, y_shift)],
            dim=0,
        )
        return out
