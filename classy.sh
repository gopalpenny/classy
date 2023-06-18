#!/bin/bash

source /Users/gopal/mambaforge/etc/profile.d/conda.sh

# This line activates the conda environment
conda activate dsgeom

# This line changes into the "classy" directory
cd /Users/gopal/Projects/ml/classy

# This line runs the "Overview.py" script
streamlit run Overview.py