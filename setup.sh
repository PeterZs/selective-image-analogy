#!/bin/bash
# Install the package. The SIA training data is read from the Hugging Face Hub
# (selective-image-analogy/SIA-Multi-Edit-Analogy) and cached on first use.
set -e

pip install -r requirements.txt
pip install -e .
