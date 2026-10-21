# Execution policy

User instruction recorded 2026-10-21: design and edit code locally; execute experiments on the 5090 server.

- All new experiment code, configuration, reports, and data belong under this repository. Do not edit the other paper directories.
- Local work is limited to reading, planning, source/configuration edits, Git operations, and viewing/downloading reports. Do not run local simulation, model checks, preprocessing, training, or evaluation scripts.
- Default execution host is SSH alias `5090`, an existing Windows workstation connection. Use the configured SSH identity and strict host-key checking; do not disable verification.
- Run data generation, native compilation, dependency/environment setup, tests, preprocessing, training, and evaluation on 5090. A model configuration change is unverified until checked there.
- Check current GPU utilization/processes before launching a job. Do not stop unrelated processes. Keep experiment jobs and outputs in a dedicated project directory and record the exact Git revision, configuration, seed, and data revision.
- GitHub stores source, configurations, Markdown, and small result summaries. Hugging Face stores dataset shards and model checkpoints. Resolve repository names from user-provided URLs rather than inventing remote destinations.
- Credentials must come from the relevant app connection, SSH authentication, or user-configured server credential/environment facilities. Never ask the user to paste tokens into chat, store them in Git, or print their values.
- The user explicitly chose public experimental code on 2026-10-21. Data, checkpoints, native solver dependencies and server access details stay private. Authentication is configured on the designated server.
- User-facing results must use MSE by default (preference recorded 2026-10-21), not relative L2 errors or percentage waveform errors. Report roof and individual floors separately, with displacement MSE in m^2 and acceleration MSE in (m/s^2)^2. For aggregate MSE, average over valid time/floors within each case, then average cases equally; exclude padding and state the split and DS subset. Do not combine channels with different units into one reported MSE. Preserve historical metrics/checkpoints and distinguish reporting changes from changes to training losses or selection criteria.
- Each experimental idea must have its own `exp/expN-short-name/` directory, `expN.md`, copied `codes/`, `configs/`, `figures/`, `results/`, and ignored `runs/`. Preserve completed source versions; a new scientific variation gets the next experiment number. Record design, exact source revision/hashes, data references, configuration, failures and results. Keep the same layout locally and on the execution host. Shared immutable datasets may be referenced rather than duplicated. Never import mutable live stage1 code into a new experiment; vendor the required source into its codes folder.

