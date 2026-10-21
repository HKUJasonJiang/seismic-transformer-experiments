# Remote execution

All simulation, preprocessing, model checks, training and evaluation run on the user-designated RTX 5090 server. Local work is limited to design, source edits, Git, transfer and report viewing.

## Public and private files

- Public GitHub: experimental source, configurations, OKR and small result summaries.
- Private storage: simulation data, native solver runtime, legacy parameter generator, checkpoints and full predictions.
- Keep hostnames, absolute server paths and credentials out of public deployment documentation.
- Record code revision, dataset manifest hash, configuration and seed in each run. Keep dataset repository identity and access configuration in the private runtime environment.

## Execution order

Run the following from the experiment checkout on the designated server:

```powershell
python stage1/scripts/check_model.py
python stage1/scripts/generate_fit_data.py
python stage1/scripts/train_poc.py
python stage1/scripts/train_poc.py --config configs/poc_refine.json
python stage1/scripts/train_poc.py --config configs/poc_balanced.json
```

The initial fit, lower-learning-rate refinement and per-case-balanced refinement use separate run IDs. Warm starts record the source checkpoint hash and source epoch. Previous results remain available.

The 24-case dataset shares one base earthquake waveform. It measures training reconstruction; independent-waveform evaluation remains a later OKR requirement.
