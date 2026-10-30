# Diffusion MRI (DTI) pipeline with DIPY

A small, reproducible pipeline to process diffusion-weighted MRI (dw-MRI) data with
[DIPY](https://dipy.org): from raw DWI to diffusion tensor maps, plus a simple
unsupervised machine-learning step on the resulting microstructural features.

## What it does

1. **Loads** a dw-MRI dataset (DWI volume + b-values + b-vectors).
2. **Builds the gradient table** and a **brain mask** (median Otsu on the b0 volumes).
3. **Fits the diffusion tensor model (DTI)** voxel-wise.
4. **Computes scalar maps**: fractional anisotropy (FA), mean, axial and radial diffusivity (MD, AD, RD) and colour-coded FA (principal diffusion direction).
5. **Clusters tissue** with k-means on standardised (FA, MD) features, relabelled as white-matter-like (high FA), grey-matter-like and CSF-like (high MD).
6. **Saves** NIfTI maps, figures (`dti_maps.png`, `fa_md_features.png`) and a summary table.

## Run it

```bash
pip install -r requirements.txt
python dti_pipeline.py
```

Data: the **Stanford HARDI** dataset (one healthy adult brain, 150 gradient directions, b = 2000 s/mm²), downloaded automatically by DIPY on the first run (~90 MB).

Outputs are written to `results/stanford_hardi/`.

## Results (Stanford HARDI)

FA highlights the corpus callosum and major white-matter tracts, MD is highest in the
ventricles (free water), and colour-FA shows the expected orientations (red: left–right,
green: anterior–posterior, blue: superior–inferior).

| Cluster | Voxels | Mean FA | Mean MD (x10⁻³ mm²/s) |
|---|---|---|---|
| CSF-like | 41,172 | 0.071 | 1.680 |
| GM-like | 123,644 | 0.143 | 0.767 |
| WM-like | 52,221 | 0.504 | 0.596 |

**Quality control.** A first run produced a "CSF" cluster of only 40 voxels with
MD ≈ 74 mm²/s, which is physically impossible (free water ≈ 3.0 × 10⁻³ mm²/s at body
temperature). 
    These were voxels where the tensor fit failed (low signal /  artefacts) and they dominated the k-means.

    Voxels with MD ≥ 3.5 × 10⁻³ mm²/s are now excluded before clustering (442 voxels, 0.20 % of the brain mask).

**Limitations.** K-means on two DTI features is only a rough tissue proxy: partial-volume voxels at tissue boundaries lower the mean MD of the CSF cluster, and a small group of noisy voxels (high FA with high MD, or FA ≈ 1 with MD ≈ 0, mostly at the mask edge) would need better masking, denoising (e.g. Patch2Self / MP-PCA) or a more robust fit (RESTORE).

## Next steps

- Denoise the DWI  and use a robust tensor fit (RESTORE).

- Compare DTI with more advanced microstructure models (DKI, NODDI/free-water) available in DIPY.
- Register the maps to a standard space (ANTs / FSL) for group-level statistics.
- Replace k-means with supervised models trained on labelled tissue or patient groups.

## Author

Lucía Fernández Fernández. Biomedical Engineer (Universidad de Málaga), MSc student in Computational Biomedical Engineering (Universitat Pompeu Fabra).
