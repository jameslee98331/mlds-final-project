#!/bin/bash
#PBS -N skew_norm_single_2d
#PBS -m be
#PBS -q jumbo

cd /home/ma/w/wkl16/coding/mlds-final-project/Research/single_component
/usr/local/bin/julia < skew_norm_single_2d.jl > skew_norm_single_2d.out
