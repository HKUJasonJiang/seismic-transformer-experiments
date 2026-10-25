"""Where in frequency do the stage-2 acceleration (and displacement) errors sit? Execution host only.

For the shown seed of each main variant on the unseen-record part: one-sided FFT over the full 66 s of every
valid floor; per frequency band, the predicted/true energy ratio (below 1 means the model attenuates that band),
the share of total error energy, and the band's contribution to the all-floor MSE (Parseval). Also checks for
patch-boundary artifacts at multiples of 1/1.2 s. Writes results/stage2/spectral_check.json and
figures/stage2/spectra.png.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP/'codes'))
from common import read, write  # noqa: E402
from datagen.shards import load_part  # noqa: E402

BANDS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 15), (15, 25.01)]
COLORS = {'truth': '#243447', 'exp5_decoder': '#2b6cb0', 'cross_attention': '#e16a2d'}


def main():
    conf = read(EXP/'configs/data.json')
    shown = read(EXP/'results/stage2/report_data.json')['shown_seed']
    cases, items, _ = load_part(conf, 'unseen_record')
    truth = {c['id']: it for c, it in zip(cases, items)}
    rate = conf['sample_rate_hz']
    n_t = conf['total_points']
    freq = np.fft.rfftfreq(n_t, 1/rate)
    out = {'frequency_resolution_hz': float(freq[1]), 'bands_hz': BANDS, 'variants': {}}
    spectra = {}
    for v in ('exp5_decoder', 'cross_attention'):
        seed = shown[v]['seed']
        with np.load(EXP/'runs/stage2'/v/f'seed{seed}'/'unseen_record_predictions.npz', allow_pickle=False) as f:
            pu, pa, ids = f['displacement'], f['acceleration'], f['case_ids']
        result = {}
        for key, pred_all in (('acceleration', pa), ('displacement', pu)):
            true_e = np.zeros(len(freq))
            pred_e = np.zeros(len(freq))
            err_e = np.zeros(len(freq))
            count = 0
            for i, cid in enumerate(ids):
                it = truth[str(cid)]
                y = it[key].numpy().astype(np.float64)
                n = y.shape[1]
                p = pred_all[i, :n_t, :n].astype(np.float64)
                ty, tp = np.fft.rfft(y, axis=0), np.fft.rfft(p, axis=0)
                true_e += (np.abs(ty)**2).sum(1)
                pred_e += (np.abs(tp)**2).sum(1)
                err_e += (np.abs(tp-ty)**2).sum(1)
                count += n
            bands = {}
            for lo, hi in BANDS:
                m = (freq >= lo) & (freq < hi)
                bands[f'{lo}-{min(hi, 25):g} Hz'] = {
                    'true_energy_share': float(true_e[m].sum()/true_e.sum()),
                    'predicted_over_true_energy': float(pred_e[m].sum()/true_e[m].sum()),
                    'error_energy_share': float(err_e[m].sum()/err_e.sum()),
                    'relative_error_energy': float(err_e[m].sum()/true_e[m].sum())}
            # Patch outputs are generated 1.2 s at a time: look for error peaks at k/1.2 Hz.
            harmonics = [k/1.2 for k in range(1, 21)]
            ratio = []
            for h in harmonics:
                j = int(round(h/freq[1]))
                neighbors = np.r_[err_e[j-6:j-2], err_e[j+3:j+7]]
                ratio.append(float(err_e[j-1:j+2].mean()/neighbors.mean()))
            result[key] = {'bands': bands, 'patch_harmonic_error_ratio_median': float(np.median(ratio)),
                           'patch_harmonic_error_ratio_max': float(max(ratio))}
            spectra[(v, key)] = (true_e/count, pred_e/count, err_e/count)
        out['variants'][v] = {'seed': seed, **result}
    write(EXP/'results/stage2/spectral_check.json', out)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, key in zip(axes, ('acceleration', 'displacement')):
        smooth = np.ones(15)/15
        t_e = spectra[('exp5_decoder', key)][0]
        ax.semilogy(freq, np.convolve(t_e, smooth, 'same'), color=COLORS['truth'], lw=1.3, label='MDOF reference')
        for v in ('exp5_decoder', 'cross_attention'):
            _, p_e, e_e = spectra[(v, key)]
            ax.semilogy(freq, np.convolve(p_e, smooth, 'same'), color=COLORS[v], lw=1, label=f'{v} prediction')
            ax.semilogy(freq, np.convolve(e_e, smooth, 'same'), color=COLORS[v], lw=1, ls='--', label=f'{v} error')
        ax.set_xlim(0, 25)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(f'{key.capitalize()}: mean energy spectrum per floor, unseen records')
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(EXP/'figures/stage2/spectra.png', dpi=150)
    for v, r in out['variants'].items():
        for key in ('acceleration', 'displacement'):
            print(v, key, 'patch-harmonic error ratio median %.2f max %.2f' % (r[key]['patch_harmonic_error_ratio_median'], r[key]['patch_harmonic_error_ratio_max']))
            for band, b in r[key]['bands'].items():
                print('   %-10s true share %5.1f%%  pred/true energy %.2f  error share %5.1f%%  relative error %.2f' % (
                    band, b['true_energy_share']*100, b['predicted_over_true_energy'], b['error_energy_share']*100, b['relative_error_energy']))
    print('SPECTRAL_CHECK_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
