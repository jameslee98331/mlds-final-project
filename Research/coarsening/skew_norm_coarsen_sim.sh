#!/bin/bash
#PBS -N skew_norm_coarsen_sim
#PBS -m be
#PBS -q jumbo

cd /home/ma/w/wkl16/coding/mlds-final-project/Research/coarsening
/usr/local/bin/julia < skew_norm_coarsen_sim.jl > skew_norm_coarsen_sim.out
