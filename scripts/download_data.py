"""Download the raw cBioPortal tables used in the paper.

Files are taken from the public cBioPortal datahub
(https://github.com/cBioPortal/datahub, studies brca_tcga, kirc_tcga,
lihc_tcga and luad_tcga) and stored under data/raw/<cohort>/.

The datahub is updated over time; if a file has changed since the analysis,
results may differ slightly from those reported in the paper.

Usage:
    python scripts/download_data.py            # all four cohorts
    python scripts/download_data.py TCGA-LUAD  # one cohort
"""
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rrgef.data import COHORT_MODALITIES, RAW_DIR, RAW_FILES  # noqa: E402

BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
STUDIES = {
    "TCGA-BRCA": "brca_tcga",
    "TCGA-KIRC": "kirc_tcga",
    "TCGA-LIHC": "lihc_tcga",
    "TCGA-LUAD": "luad_tcga",
}


def main():
    cohorts = sys.argv[1:] or list(STUDIES)
    for cohort in cohorts:
        out_dir = os.path.join(RAW_DIR, cohort)
        os.makedirs(out_dir, exist_ok=True)
        for mod in COHORT_MODALITIES[cohort]:
            name = RAW_FILES[mod]
            dest = os.path.join(out_dir, name)
            if os.path.exists(dest):
                print(f"[skip] {dest}")
                continue
            url = f"{BASE}/{STUDIES[cohort]}/{name}"
            print(f"[get ] {url}")
            urllib.request.urlretrieve(url, dest)
    print("done")


if __name__ == "__main__":
    main()
