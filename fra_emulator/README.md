fra_emulator
===============

Tiny TensorFlow-based emulator for CLASS matter power spectra at z=0.

Files
- `emulator.py`: emulator class with TF/sklearn/numpy fallbacks
- `generate_synthetic.py`: toy P(k) generator for quick training
- `train.py`: example trainer that saves a model under `models/demo`
- `test_emulator.py`: a smoke test for local verification

Usage
-----
You can train a demo model quickly (no CLASS required):

```bash
python -m fra_emulator.train
```

Then load and predict from Python:

```python
from fra_emulator.emulator import PowerSpectrumEmulator
em = PowerSpectrumEmulator.load('fra_emulator/models/demo')
params = [0.12, 67.0, 3.05, 0.965, 0.06]
pk = em.predict(params)
```

Notes
-----
This is a lightweight demo. For production-grade emulation you should
generate training data from CLASS over a well-sampled parameter space,
add proper data normalization, cross-validation, and uncertainty quantification.
