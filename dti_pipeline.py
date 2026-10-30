"""
Diffusion MRI (dMRI) processing pipeline with DIPY.

Steps
1. Load a diffusion-weighted MRI dataset (DWI + b-values + b-vectors).
2. Build the gradient table and a brain mask (median Otsu on the b0 volumes).
3. Fit the diffusion tensor model (DTI).
4. Compute scalar maps: FA, MD, AD, RD and the colour-coded FA (principal direction).
5. Unsupervised tissue clustering (k-means on FA + MD) as a simple ML step:
   white matter (high FA), grey matter (low FA, low MD) and CSF (high MD).
6. Save NIfTI maps, figures and a summary table.

Data: Stanford HARDI dataset (one healthy brain, 150 gradient directions, b = 2000 s/mm²),
downloaded automatically by DIPY the first time (~90 MB).

Usage
    python dti_pipeline.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.dti import TensorModel, color_fa, fractional_anisotropy
from dipy.segment.mask import median_otsu
from sklearn.cluster import KMeans


def load_data():
    """Download (first run only) and load the Stanford HARDI dataset."""
    from dipy.data import get_fnames

    dwi, bval, bvec = get_fnames(name="stanford_hardi")
    img = nib.load(str(dwi))
    bvals, bvecs = read_bvals_bvecs(str(bval), str(bvec))
    return img.get_fdata(dtype=np.float32), img.affine, bvals, bvecs, "stanford_hardi"


def brain_mask(data, bvals, b0_threshold=50):
    """Median-Otsu brain mask computed on the b0 volumes."""
    b0_idx = np.where(bvals <= b0_threshold)[0]
    masked, mask = median_otsu(
        data, vol_idx=b0_idx, median_radius=3, numpass=1, dilate=2
    )
    return masked, mask


def fit_dti(data, gtab, mask):
    model = TensorModel(gtab)
    fit = model.fit(data, mask=mask)
    fa = np.clip(np.nan_to_num(fractional_anisotropy(fit.evals)), 0, 1)
    md = np.nan_to_num(fit.md)
    ad = np.nan_to_num(fit.ad)
    rd = np.nan_to_num(fit.rd)
    rgb = color_fa(fa, fit.evecs)
    return {"FA": fa, "MD": md, "AD": ad, "RD": rd}, rgb


MD_MAX = 3.5e-3  # mm²/s, just above free water at body temperature (~3.0e-3)


def valid_voxels(md, mask):
    """Voxels inside the mask with a physiologically plausible tensor fit.

    A few voxels give non-physical fits (MD far above free water, FA = 0), usually
    because of low signal or artefacts. They are excluded from the clustering so
    they do not distort it.
    """
    return mask & (md > 0) & (md < MD_MAX)


def tissue_clustering(fa, md, valid, n_clusters=3, seed=0):
    """K-means on standardised (FA, MD) features of the valid voxels.

    Clusters are relabelled so that 1 = CSF (highest MD), 3 = WM (highest FA)
    and 2 = GM (the remaining one). Excluded voxels keep label 0.
    """
    mask = valid
    X = np.column_stack([fa[mask], md[mask]])
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(Xz)
    labels = km.labels_
    centres = np.array([X[labels == k].mean(0) for k in range(n_clusters)])
    csf = int(np.argmax(centres[:, 1]))
    wm = int(np.argmax(np.where(np.arange(n_clusters) == csf, -np.inf, centres[:, 0])))
    order = {csf: 1, wm: 3}
    for k in range(n_clusters):
        order.setdefault(k, 2)
    seg = np.zeros(mask.shape, dtype=np.uint8)
    seg[mask] = np.vectorize(order.get)(labels)
    names = {1: "CSF-like", 2: "GM-like", 3: "WM-like"}
    stats = []
    for lab in (1, 2, 3):
        sel = seg == lab
        if sel.any():
            stats.append((names[lab], int(sel.sum()), fa[sel].mean(), md[sel].mean()))
    return seg, stats


def save_figures(maps, rgb, seg, mask, outdir, name):
    """`mask` here is the set of valid voxels used for the clustering."""
    z = maps["FA"].shape[2] // 2
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.5))
    panels = [
        (maps["FA"][:, :, z].T, "Fractional anisotropy (FA)", "gray", (0, 1)),
        (maps["MD"][:, :, z].T * 1e3, "Mean diffusivity (MD, x10⁻³ mm²/s)", "magma", None),
        (np.transpose(rgb[:, :, z], (1, 0, 2)), "Colour FA (principal direction)", None, None),
        (seg[:, :, z].T, "k-means tissue clusters", "viridis", (0, 3)),
    ]
    for a, (img, title, cmap, lim) in zip(ax, panels):
        kw = {"origin": "lower"}
        if cmap:
            kw["cmap"] = cmap
        if lim:
            kw["vmin"], kw["vmax"] = lim
        im = a.imshow(img, **kw)
        a.set_title(title, fontsize=10)
        a.axis("off")
        if cmap and title.startswith(("Fractional", "Mean")):
            fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"DTI maps – {name} (axial slice {z})")
    fig.tight_layout()
    fig.savefig(outdir / "dti_maps.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(maps["FA"][mask], bins=60, color="steelblue")
    ax[0].set_xlabel("FA")
    ax[0].set_ylabel("Voxels")
    ax[0].set_title("FA distribution inside the brain mask")
    sc = ax[1].scatter(
        maps["FA"][mask], maps["MD"][mask] * 1e3, c=seg[mask], s=2, cmap="viridis", vmin=0, vmax=3
    )
    ax[1].set_xlabel("FA")
    ax[1].set_ylabel("MD (x10⁻³ mm²/s)")
    ax[1].set_title("Feature space used for k-means")
    fig.tight_layout()
    fig.savefig(outdir / "fa_md_features.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    data, affine, bvals, bvecs, name = load_data()
    print(f"Loaded {name}: shape {data.shape}, b-values {sorted(set(np.round(bvals, -1)))}")
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)

    data_m, mask = brain_mask(data, bvals)
    print(f"Brain mask: {int(mask.sum())} voxels")

    maps, rgb = fit_dti(data_m, gtab, mask)
    valid = valid_voxels(maps["MD"], mask)
    n_bad = int(mask.sum() - valid.sum())
    print(f"Excluded {n_bad} voxels with non-physical tensor fits ({100 * n_bad / mask.sum():.2f}% of the mask)")
    seg, stats = tissue_clustering(maps["FA"], maps["MD"], valid)

    outdir = Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)
    for k, v in maps.items():
        nib.save(nib.Nifti1Image(v.astype(np.float32), affine), outdir / f"{k}.nii.gz")
    nib.save(nib.Nifti1Image(np.array(255 * rgb, dtype=np.uint8), affine), outdir / "colorFA.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine), outdir / "tissue_kmeans.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), outdir / "brain_mask.nii.gz")
    save_figures(maps, rgb, seg, valid, outdir, name)

    lines = ["| Cluster | Voxels | Mean FA | Mean MD (x10⁻³ mm²/s) |", "|---|---|---|---|"]
    for n, v, f, m in stats:
        lines.append(f"| {n} | {v} | {f:.3f} | {m * 1e3:.3f} |")
    summary = "\n".join(lines)
    (outdir / "summary.md").write_text(summary + "\n")
    print(f"Mean FA (valid voxels): {maps['FA'][valid].mean():.3f}")
    print(summary)
    print(f"Outputs saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
