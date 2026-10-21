# exp3 execution

Execute only on the configured 5090 server. From this experiment directory:

```text
python codes/run.py --smoke-only
python codes/run.py
python codes/report.py
python codes/audit_and_export.py
python codes/archive.py
```

`run.py --protocol multiwave|period|stories` selects one protocol. Data are immutable references to the existing stage1 dataset. No test manifest is opened. Full-batch training uses the fixed budgets in configs/run.json; checkpoint selection uses training objectives only.

Training was performed with the source revision recorded in every result file. `inference.py` and `audit_and_export.py` were added after training as acceptance/export tools; they do not change trained weights. The final audit verifies all originally hashed source files, checkpoint hashes, equal A/B initialization, saved waveform MSE, frozen B, and label-free roof querying.

Completed results are immutable. To repeat training with any new source/configuration, create the next experiment directory instead of overwriting this version. Historical source snapshots and actual run provenance remain available in Git and the private archive.

The deployment files are `runs/<protocol>-student/inference.pt`. Load with `inference.load_model`, then call `inference.predict` using ground motion, the four physical attributes, structure type, floor queries and a time mask. There is no response-label argument and no C in this checkpoint. Outputs are relative displacement in metres and absolute acceleration in m/s², preserving the calibrated solver's convention.

`archive.py` requires the already configured server Hugging Face credentials and HF_REPO_ID pointing to the user's private dataset repository. Never place credentials in source or configuration.
