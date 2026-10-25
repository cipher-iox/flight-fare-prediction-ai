# Models Directory

This directory stores the trained model bundle produced by the modeling notebook.

## Contents

| File | Description |
|------|-------------|
| `best_model.pkl` | Best-performing ML model bundle (saved after 10-Fold CV) |

## Model Bundle Specification

The pickle file contains a dictionary with:
```python
{
    'model': fitted_model,
    'le_dict': encoders_dictionary,
    'features': feature_names_list,
    'target': 'Price',
    'model_name': 'Best Estimator Name',
}
```
