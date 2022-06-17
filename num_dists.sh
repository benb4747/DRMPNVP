#PBS -l ncpus=40
#PBS -N ambiguity_sets_Poisson
#PBS -o Output
#PBS -j oe
#PBS -m ae
#PBS -M b.black1@lancaster.ac.uk

# wherever the script is
cd ~/PDRO_Newsvendor/
## Set num_threads to the same number as you set ncpus
export num_threads=40
### Make sure the following is before you run Rscript (or other language)
### One of these packages will control
### how many cores linear algebra uses up
export MKL_NUM_THREADS=$num_threads ###Limits Intel math kernel
export OPENBLAS_NUM_THREADS=$num_threads ### Limits OPENBLAS, this is the most important one
export MC_CORES=$num_threads ###Limits some packages in R
export OMP_NUM_THREADS=$num_threads ### Limits OpenMP
export NUMEXPR_NUM_THREADS=$num_threads ### Limits NUMEXR in p

python3 num_dists_poisson.py $PBS_ARRAYID