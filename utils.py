"""Util functions."""
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn


def apply_padding(input: np.ndarray, padding: float):
    input_size = input.shape[0]

    input_ = np.full((input_size + padding, input_size + padding), np.nan)
    offset = padding // 2
    input_[offset:offset + input_size, offset:offset + input_size] = input
    return input_


def generate_input(input_size: int, padding: int = 0):
    if padding > 0:
        input = np.full((input_size + padding, input_size + padding), np.nan)
        offset = padding // 2
        input[offset:offset + input_size, offset:offset + input_size] = \
            np.random.random((input_size, input_size))
        return input

    return np.random.random((input_size, input_size))


def generate_kernel(kernel_size: int):
    return np.random.random((kernel_size, kernel_size))


def apply_convolution(input: np.ndarray, kernel: np.ndarray, stride: int = 1):
    input = np.nan_to_num(input)
    with torch.no_grad():
        conv = nn.Conv2d(1, 1, kernel_size=kernel.shape, stride=stride, bias=False)
        conv.weight = nn.Parameter(torch.Tensor(kernel[None, None, :, :]))
        output = conv(torch.Tensor(input[None, None, :, :]))
        return output.numpy()[0][0]


def apply_transposed_convolution(input: np.ndarray, kernel: np.ndarray, stride: int = 1):
    input = np.nan_to_num(input)
    with torch.no_grad():
        conv = nn.ConvTranspose2d(1, 1, kernel_size=kernel.shape, stride=stride, bias=False)
        conv.weight = nn.Parameter(torch.Tensor(kernel[None, None, :, :]))
        output = conv(torch.Tensor(input[None, None, :, :]))
        return output.numpy()[0][0]


def apply_pooling(input: np.ndarray, kernel: np.ndarray, stride: int = 1, pooling_op: str = 'mean'):
    input = np.nan_to_num(input)
    with torch.no_grad():
        if pooling_op == 'mean':
            pool = nn.AvgPool2d(kernel.shape, stride=stride)
        elif pooling_op == 'max':
            pool = nn.MaxPool2d(kernel.shape, stride=stride)
        output = pool(torch.Tensor(input[None, None, :, :]))
        return output.numpy()[0][0]


def plot_array(array: np.ndarray, vmin: float = None, vmax: float = None, grid: bool = True):
    fig = plt.figure()
    plt.imshow(array, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.gcf().set_facecolor('gray')

    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, array.shape[0]-0.5, 1))
    ax.set_yticks(np.arange(-0.5, array.shape[0]-0.5, 1))
    if grid:
        ax.grid(color='gray', linestyle='-', linewidth=1)
    plt.tight_layout()
    return fig
