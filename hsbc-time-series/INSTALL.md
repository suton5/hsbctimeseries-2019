# Create environment
conda create -n ds-training python=3 jupyter numpy pandas matplotlib seaborn python-graphviz scikit-learn pandas-datareader 

## Activate environment
source activate ds-training
## or
conda activate ds-training

## install prophet
conda install -c conda-forge fbprophet