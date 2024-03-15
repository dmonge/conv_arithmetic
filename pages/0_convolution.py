"""Streamlit page for exploring convolution arithmetic."""
import streamlit as st

import numpy as np


from utils import generate_input
from utils import generate_kernel
from utils import compute_output
from utils import plot_array


def _compute_output_size(input_size: int, padding: int, kernel_size: int, stride: int):
    return (input_size + padding - kernel_size) // stride + 1


def render_controls():
    input_size = st.number_input('Input size', min_value=1, value=5)
    padding = st.number_input('Padding', min_value=0, value=0)
    kernel_size = st.number_input('Kernel size', min_value=1, value=3)
    stride = st.number_input('Stride', min_value=1, value=1)

    return input_size, kernel_size, stride, padding


def main():
    st.title('Convolution arithmetic')

    st.markdown('Given:')
    st.markdown(' - an input array with size _i_,')
    st.markdown(' - which has been padded with _p_ rows/columns,')
    st.markdown(' - a kernel with size _k_,')
    st.markdown(' - applied with stride _s_,')
    st.markdown('then, the size of the convolution output, _o_, is:')
    st.latex(r'o = \left\lfloor (i + p - k) / s \right\rfloor + 1')

    # controls
    st.subheader('Controls')
    input_size, kernel_size, stride, padding = render_controls()

    # data
    input = generate_input(input_size, padding=padding)
    kernel = generate_kernel(kernel_size)
    output = compute_output(input, kernel, stride)
    vmin = min(np.nanmin(input), np.min(kernel), np.min(output))
    vmax = max(np.nanmax(input), np.max(kernel), np.max(output))
    st.write(f'Output size: {_compute_output_size(input_size, padding, kernel_size, stride)}')

    # plots
    st.subheader('Convolution')
    c1, c2, c3 = st.columns([1, 1, 1])

    c1.text('Input')
    c1.pyplot(plot_array(input, vmin=vmin, vmax=vmax))

    c2.text('Kernel')
    c2.pyplot(plot_array(kernel, vmin=vmin, vmax=vmax))

    c3.text('Output')
    c3.pyplot(plot_array(output, vmin=vmin, vmax=vmax))


main()
