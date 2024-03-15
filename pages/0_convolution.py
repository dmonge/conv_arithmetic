"""Streamlit page for exploring convolution arithmetic."""
import streamlit as st

import numpy as np


from utils import generate_input
from utils import generate_kernel
from utils import compute_output
from utils import plot_array


def render_controls():
    st.title('Convolution arithmetic')
    input_size = st.number_input('Input size', min_value=1, value=5)
    padding = st.number_input('Padding', min_value=0, value=0)
    kernel_size = st.number_input('Kernel size', min_value=1, value=3)
    stride = st.number_input('Stride', min_value=1, value=1)

    return input_size, kernel_size, stride, padding


def main():
    # controls
    input_size, kernel_size, stride, padding = render_controls()

    # data
    input = generate_input(input_size, padding=padding)
    kernel = generate_kernel(kernel_size)
    output = compute_output(input, kernel, stride)
    vmin = min(np.nanmin(input), np.min(kernel), np.min(output))
    vmax = max(np.nanmax(input), np.max(kernel), np.max(output))

    # plots
    st.subheader('Convolution')
    c1, c2, c3 = st.columns([1, 1, 1])

    c1.text('Input')
    c1.pyplot(plot_array(input, container=c1, vmin=vmin, vmax=vmax))

    c2.text('Kernel')
    c2.pyplot(plot_array(kernel, container=c2, vmin=vmin, vmax=vmax))

    c3.text('Output')
    c3.pyplot(plot_array(output, container=c3, vmin=vmin, vmax=vmax))


main()
