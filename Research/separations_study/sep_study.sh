#!/bin/bash
#PBS -N sep_study
#PBS -m be
#PBS -q large256

cd /home/ma/w/wkl16/coding/mlds-final-project/Research/separations_study
/usr/local/bin/julia < sep_study.jl > sep_study.out
