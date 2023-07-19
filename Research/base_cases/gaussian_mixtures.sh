#!/bin/bash
#PBS -N gaussian_mixtures
#PBS -m be
#PBS -q large256

cd /home/ma/w/wkl16/coding/mlds-final-project/Research/base_cases
/usr/local/bin/julia < gaussian_mixtures.jl > gaussian_mixtures_sim.out
