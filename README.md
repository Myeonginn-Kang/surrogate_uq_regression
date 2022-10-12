# surrogate_uq_regression
PyTorch Implementation for surrogate approach to uncertainty quantification of neural networks for regression


## Components
- **data/*** - data files
- **model/*** - trained model files
- **data_stats.py** - summary statistics of datasets
- **train_models.py** - model training
- **surrogates.py** - Proposed surrogates for uncertainty quantification: Input perturbation, Gradient norm, MC-dropout, 'Knowledge distillation', 'Ensemble'
- **calculate_score.py** - Calculating the uncertainty score from the proposed surrogate
- **example.py** - Example code for obtaining the uncertainty scroe of query instances
- **utils.py**

## Dependencies
- **Python**
- **PyTorch**
- **NumPy**
- **scikit-learn**

