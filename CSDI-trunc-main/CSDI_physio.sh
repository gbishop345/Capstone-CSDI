#!/bin/bash
#SBATCH --job-name=csdi_physio_job   # Job name
#SBATCH --ntasks=1                                    
#SBATCH --gres=gpu:1                 
#SBATCH --partition=gpu-v100         
#SBATCH --time=04:00:00              
#SBATCH --output=csdi_physio_train_%j.log  

# Activate the virtual environment
source /data/projects/isabelle.luebbert/csdi_env/bin/activate  # Adjust the path to your environment

# Navigate to the project directory
cd /scratch/isabelle.luebbert/CSDI-trunc-main

# Run training and imputation for the healthcare dataset
python exe_physio.py --testmissingratio 0.1 --nsample 100

# Deactivate the virtual environment (optional)
deactivate
