"""Windows only: export legacy ten-parameter story materials for every (type, stories, fortification) in configs/data.json.

The archived parameter generator is a Windows executable (matcodeinput=3: parameter export only; no responses
are used). Mirrors stage1/scripts/five_story_preview.py baseline() for each combination.
  python codes/datagen/export_materials.py --out <folder>
Writes <folder>/materials.json; copy that file to the execution host and point configs/data.json at it.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
PROJECT = EXP.parents[2]
SOURCE_EXE = PROJECT/'3-Data/MDOF_BenchMark/Flexural_Shear_0.1.exe'
SOURCE_INPUTS = PROJECT/'mdof-workbench/examples/validation/legacy-models/inputs'
HEADER = ('Blg_LB Stories height Year T1 T2 Area IM Site_CD Site_GP YieldOver PeakOver YieldStiff PGA dt Ttime '
          'StrucType matcodeinput OutputCtrl BlgCondition MasonryWallRatio Width')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(folder, b, stories, intensity, structure_type):
    result = folder/'Results/matcode1.txt'
    if not result.exists():
        for name in ('inputs', 'GMs', 'Results'):
            (folder/name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_EXE, folder/'legacy.exe')
        shutil.copy2(SOURCE_INPUTS/'config.txt', folder/'inputs/config.txt')
        shutil.copy2(SOURCE_INPUTS/'DensifyingConfig.txt', folder/'inputs/DensifyingConfig.txt')
        lines = (folder/'inputs/config.txt').read_bytes().decode('gbk').splitlines()
        values = lines[1].split()
        values[0], values[1], values[3], values[4] = str(b['damping_ratio']), '0.02', str(b['mass_per_area_kg_m2']), '0'
        lines[1] = '\t'.join(values)
        (folder/'inputs/config.txt').write_bytes(('\r\n'.join(lines) + '\r\n').encode('gbk'))
        (folder/'inputs/Dir.txt').write_text('./Results\npreview\n', encoding='ascii')
        (folder/'inputs/EQ_List.txt').write_text('record\n', encoding='ascii')
        row = [1, stories, stories*b['story_height_m'], b['year'], -1, -1, b['floor_area_m2'], intensity,
               b['site_category'], b['site_group'], -1, -1, -1, 1, 0.02, 1, structure_type, 3, 111000, -1, -1, -1]
        (folder/'inputs/BlgAttributes.txt').write_text('1\n' + HEADER + '\n' + ' '.join(map(str, row)) + '\n', encoding='ascii')
        (folder/'GMs/record.txt').write_text('51\n' + ''.join(f'{i*.02:.4f}\t0\n' for i in range(51)), encoding='ascii')
        p = subprocess.run([str(folder/'legacy.exe')], cwd=folder, input=b'\n', capture_output=True, timeout=60,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        (folder/'stdout.log').write_bytes(p.stdout)
        (folder/'stderr.log').write_bytes(p.stderr)
        if p.returncode or not result.exists():
            raise RuntimeError(f'Material export failed, exit={p.returncode}: {folder}')
    raw = result.read_text(encoding='ascii').splitlines()
    if int(raw[0].strip()) != stories:
        raise ValueError(f'Exported story count is not {stories}: {folder}')
    material = [[float(v.replace('D', 'E')) for v in line.replace(',', ' ').split()]
                for line in raw if len(line.replace(',', ' ').split()) == 10]
    if len(material) != stories:
        raise ValueError(f'Cannot parse {stories} ten-parameter rows: {folder}')
    return material, {'material_export_sha256': sha(result),
                      'input_sha256': {p.name: sha(p) for p in sorted((folder/'inputs').iterdir())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    conf = json.loads((EXP/'configs/data.json').read_text(encoding='utf-8'))
    b, s = conf['building'], conf['structures']
    out = Path(args.out)
    stories = sorted(set(s['training_stories'] + s['interpolation_stories'] + s['extrapolation_stories']))
    table = {}
    for n in stories:
        for im in s['fortification_levels']:
            key = f'type{b["structure_type"]}_N{n:02d}_IM{im}'
            material, provenance = export(out/key, b, n, im, b['structure_type'])
            table[key] = {'stories': n, 'fortification': im, 'structure_type': b['structure_type'],
                          'material': material, 'provenance': provenance}
            print(key, 'E0[0]=%.4g sy0[0]=%.4g' % (material[0][0], material[0][1]), flush=True)
    result = {'generator_executable_sha256': sha(SOURCE_EXE), 'mode': 'matcodeinput=3 parameter export only',
              'building': b, 'rows': 'bottom story first; columns E0 (N/m), sy0 (N), then eight dimensionless parameters',
              'materials': table}
    (out/'materials.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('MATERIALS_EXPORTED', len(table), sha(out/'materials.json'), flush=True)


if __name__ == '__main__':
    main()
