# surrogate_uq_regression
Source code for the paper: Surrogate approach to uncertainty quantification of neural networks for regression

## Components
- **data/*** - data files
- **model/*** - trained model files
- **regression_model.py** - training and inference functions for the given regression network
- **surrogates.py** - script for the proposed surrogates for uncertainty quantification: Input perturbation, Gradient norm, MC-dropout, Knowledge distillation, Ensemble
- **uncertainty_score.py** - functions for calculating the uncertainty score from the proposed surrogate
- **main.py** - script for overall running code
- **data_stats.py** - summary statistics of datasets
- **utils.py**

## Dependencies
- **Python**
- **PyTorch**
- **NumPy**
- **scikit-learn**

