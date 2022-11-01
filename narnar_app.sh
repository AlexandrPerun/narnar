#!/bin/bash

source /home/peoly/miniconda3/etc/profile.d/conda.sh
cd /mnt/data/Alex_tmp/narnar/
conda activate narnar
streamlit run app.py
