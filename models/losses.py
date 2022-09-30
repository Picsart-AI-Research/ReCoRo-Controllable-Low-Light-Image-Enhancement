import os
from pathlib import Path
from typing import Sequence, Union
from hydra.utils import get_original_cwd

import torch
from torch import nn
import torch.nn.functional as F
from data.transform import get_denorm_transform
from models.auxiliary import Vgg16



def get_reduction(reduction):
    if reduction == "none":
        def reduce(x):
            return x
    elif reduction == "mean":
        def reduce(x):
            return torch.mean(x)
    elif reduction == "instance":
        def reduce(x):
            return torch.mean(x, dim=[1, 2, 3])
    else:
        raise "Unknown loss reduction"

    return reduce


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, cfg, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss and use no_sigmoid if you want to omit it.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = cfg.gan_mode
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss(reduction=cfg.loss_reduction)
        elif self.gan_mode == 'no_sigmoid':
            self.loss = nn.BCELoss(reduction=cfg.loss_reduction)
        elif self.gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(reduction=cfg.loss_reduction)
        elif self.gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(
                'gan mode %s not implemented' % self.gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, alpha=None, reduction="mean", mask=None):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        reduce = get_reduction(reduction)
        if self.gan_mode in ['lsgan', 'vanilla', 'no_sigmoid']:
            if mask is None:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
            else:
                target_tensor = mask
            loss = reduce(self.loss(prediction, target_tensor))
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                # loss = -prediction.mean()
                loss = reduce(-prediction)
            else:
                # loss = prediction.mean()
                loss = reduce(prediction)
        return loss

class PerceptualLoss(nn.Module):
    bgr_permute = [2, 1, 0]

    def __init__(self, cfg, device=None):
        super(PerceptualLoss, self).__init__()
        self.cfg = cfg
        self.denorm_transform = get_denorm_transform(cfg)
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.vgg = self.load_vgg16(cfg.vgg)
        self.loss = nn.MSELoss(reduction=cfg.loss_reduction)

    def load_vgg16(self, cfg):
        """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
        vgg = Vgg16(cfg)
        vgg.cuda()
        # vgg.cuda(device=device)
        orig_cwd = get_original_cwd() # "."
        weights_path = Path(f"{orig_cwd}/{cfg.weights_path}")

        vgg.load_state_dict(torch.load(weights_path))
        for param in vgg.parameters():
            param.requires_grad = False

        vgg.eval()
        return vgg

    def __call__(self, img, target, reduction="mean"):
        img_denorm = self.denorm_transform(img) * 255.0
        target_denorm = self.denorm_transform(target) * 255.0
        img_bgr = img_denorm[..., self.bgr_permute, :, :]
        target_bgr = target_denorm[..., self.bgr_permute, :, :]
        img_feat = self.vgg(img_bgr)
        target_feat = self.vgg(target_bgr)
        reduce = get_reduction(reduction)
        return reduce(self.loss(self.instancenorm(img_feat), self.instancenorm(target_feat)))

class SSIMLoss(nn.Module):
    """
    Computes Structual Similarity Index Measure

    Args:
        data_range: Range of the image. Typically, ``1.0`` or ``255``.
        kernel_size: Size of the kernel. Default: (11, 11)
        sigma: Standard deviation of the gaussian kernel.
            Argument is used if ``gaussian=True``. Default: (1.5, 1.5)
        k1: Parameter of SSIM. Default: 0.01
        k2: Parameter of SSIM. Default: 0.03
        gaussian: ``True`` to use gaussian kernel, ``False`` to use uniform kernel
        output_transform: A callable that is used to transform the
            :class:`~ignite.engine.engine.Engine`'s ``process_function``'s output into the
            form expected by the metric.
        device: specifies which device updates are accumulated on. Setting the metric's
            device to be the same as your ``update`` arguments ensures the ``update`` method is non-blocking. By
            default, CPU.

    Examples:
        To use with ``Engine`` and ``process_function``, simply attach the metric instance to the engine.
        The output of the engine's ``process_function`` needs to be in the format of
        ``(y_pred, y)`` or ``{'y_pred': y_pred, 'y': y, ...}``.

        ``y_pred`` and ``y`` can be un-normalized or normalized image tensors. Depending on that, the user might need
        to adjust ``data_range``. ``y_pred`` and ``y`` should have the same shape.

        .. testcode::

            metric = SSIM(data_range=1.0)
            metric.attach(default_evaluator, 'ssim')
            preds = torch.rand([4, 3, 16, 16])
            target = preds * 0.75
            state = default_evaluator.run([[preds, target]])
            print(state.metrics['ssim'])

        .. testoutput::

            0.9218971...

    .. versionadded:: 0.4.2
    """

    def __init__(
        self,
        data_range: Union[int, float],
        kernel_size: Union[int, Sequence[int]] = (11, 11),
        sigma: Union[float, Sequence[float]] = (1.5, 1.5),
        k1: float = 0.01,
        k2: float = 0.03,
        gaussian: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(SSIMLoss, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]  # type: Sequence[int]
        elif isinstance(kernel_size, Sequence):
            self.kernel_size = kernel_size
        else:
            raise ValueError("Argument kernel_size should be either int or a sequence of int.")

        if isinstance(sigma, float):
            self.sigma = [sigma, sigma]  # type: Sequence[float]
        elif isinstance(sigma, Sequence):
            self.sigma = sigma
        else:
            raise ValueError("Argument sigma should be either float or a sequence of float.")

        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")

        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")
        self.device = device
        self.gaussian = gaussian
        self.c1 = (k1 * data_range) ** 2
        self.c2 = (k2 * data_range) ** 2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma).to(device)


    def _uniform(self, kernel_size: int) -> torch.Tensor:
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size: int, sigma: float) -> torch.Tensor:
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size: Sequence[int], sigma: Sequence[float]) -> torch.Tensor:
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)


    def forward(self, y, y_pred) -> None:

        if y_pred.dtype != y.dtype:
            raise TypeError(
                f"Expected y_pred and y to have the same data type. Got y_pred: {y_pred.dtype} and y: {y.dtype}."
            )

        if y_pred.shape != y.shape:
            raise ValueError(
                f"Expected y_pred and y to have the same shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        if len(y_pred.shape) != 4 or len(y.shape) != 4:
            raise ValueError(
                f"Expected y_pred and y to have BxCxHxW shape. Got y_pred: {y_pred.shape} and y: {y.shape}."
            )

        channel = y_pred.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        y_pred = F.pad(y_pred, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        y = F.pad(y, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([y_pred, y, y_pred * y_pred, y * y, y_pred * y])
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * y_pred.size(0) : (x + 1) * y_pred.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        return 1.0 - torch.mean(ssim_idx, dim=(1, 2, 3))
