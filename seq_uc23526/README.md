# Sequential Data Section

## Notebook

Download the PyMC example notebook from:
https://github.com/pymc-devs/pymc-examples/blob/main/examples/causal_inference/excess_deaths.ipynb

After downloading, make the following edits to run with current PyMC:
1. Change all occurrences of `pm.MutableData` to `pm.Data`
2. In the `ZeroSumNormal` function, find `model.add_coord` and remove the `mutable` keyword argument

## Files

- `hmms.py` - HMM implementation code task
- `report_seq.pdf` - Sequential data report (convert to PDF for submission)

