"""Streamlit page for exploring the operators."""
from glob import glob

import numpy as np
from torchvision.io import read_image
from torchvision.transforms import v2
import streamlit as st

from utils import apply_padding
from utils import generate_kernel
from utils import apply_convolution
from utils import apply_pooling
from utils import apply_transposed_convolution
from utils import plot_array


IMAGES = {p.split('/')[-1][:-4]: p
          for p in sorted(glob('images/*.png'))}


def render_image_selection():
    image = st.selectbox('Image', IMAGES)
    size = st.selectbox('Size', [64, 128, 256])
    image = read_image(IMAGES[image])
    image = v2.Compose([
        v2.Grayscale(),
        v2.Resize(size),
    ])(image)
    image = image.numpy().squeeze()
    return image


def render_controls(operation):
    apply = st.checkbox(f'Apply {operation}', value=False)
    padding_help = 'Padding will be shown as white pixels on the input.'
    padding = st.number_input(f'Padding {operation}', min_value=0, value=0, help=padding_help, disabled=not apply)
    kernel_size = st.number_input(f'Kernel size {operation}', min_value=1, value=3, disabled=not apply)
    stride = st.number_input(f'Stride {operation}', min_value=1, value=1, disabled=not apply)

    return kernel_size, stride, padding, apply


def display_arrays(input, kernel, output, vmax=None, vmin=None):
    """Display arrays."""
    c1, c2, c3 = st.columns([1, 1, 1])
    c1.text('Input')
    c1.pyplot(plot_array(input, vmin=vmin, vmax=vmax, grid=False))
    c2.text('Kernel')
    c2.pyplot(plot_array(kernel, vmin=vmin, vmax=vmax))
    c3.text('Output')
    c3.pyplot(plot_array(output, vmin=vmin, vmax=vmax, grid=False))
    st.info(f'Output size: {output.shape[0]}')


def main():
    st.title('Operator explorer')

    # input image
    st.subheader('Input image')
    input = render_image_selection()
    st.image(input)

    st.subheader('Operations')
    # convolution
    with st.expander('Convolution'):
        kernel_size, stride, padding, apply = render_controls('convolution')

        if apply:
            # data
            input = apply_padding(input, padding)
            kernel = generate_kernel(kernel_size) * 128
            output = apply_convolution(input, kernel, stride)

            # plots
            st.divider()
            display_arrays(input, kernel, output)
            input = output

    # pooling
    with st.expander('Pooling'):
        kernel_size, stride, padding, apply = render_controls('pooling')

        if apply:
            # data
            input = apply_padding(input, padding)
            kernel = np.full((kernel_size, kernel_size), np.nan)
            output = apply_pooling(input, kernel, stride, pooling_op='max')

            # plots
            st.divider()
            display_arrays(input, kernel, output)
            input = output

    # transposed convolution
    with st.expander('Transposed convolution'):
        kernel_size, stride, padding, apply = render_controls('transposed convolution')

        if apply:
            # data
            input = apply_padding(input, padding)
            kernel = generate_kernel(kernel_size) * 128
            output = apply_transposed_convolution(input, kernel, stride)

            # plots
            st.divider()
            display_arrays(input, kernel, output)
            input = output

    # final result
    st.subheader('Result')
    st.image((input - input.min()) / (input.max() - input.min()))


main()
