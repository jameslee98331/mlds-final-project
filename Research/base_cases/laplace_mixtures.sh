#!/bin/bash
#PBS -N laplace_mixtures
#PBS -m e
#PBS -q large256

cd /home/ma/w/wkl16/coding/mlds-final-project/Research/base_cases
/usr/local/bin/julia < laplace_mixtures.jl > laplace_mixtures_sim.out
