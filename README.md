# LAB-GUARDIAN AI — Streamlit Research Demonstration

A synthetic-data proof of concept for multidisciplinary specimen screening and pre-analytical quality analysis. **Not for clinical use.**

## Deploy on Streamlit Community Cloud
1. Sign in to GitHub and create a new repository named `LAB-GUARDIAN-AI`.
2. Choose **Add file → Upload files** and upload `app.py`, `requirements.txt`, and `synthetic_specimens_1000.csv` from this folder (not the ZIP itself). Commit changes.
3. Open https://share.streamlit.io and choose **Create app → Deploy a public app from GitHub**.
4. Choose your new repository, main branch, and main file `app.py`. Click **Deploy**.
5. Try the demo cases on the first tab. Review EDA, ML evaluation and quality indicators on the other tabs.

## Run locally (optional)
`pip install -r requirements.txt`
`streamlit run app.py`

## Research limitations
The 1,000 records are synthetic. The screening rules are deliberately simplified, illustrative and unapproved. The classifier uses an 80:20 stratified split to predict synthetic decisions; evaluation does not establish clinical validity. No real patient data should be entered. Laboratory SOPs, manufacturer specifications, rule governance, independent expert annotation, retrospective testing and silent-mode prospective validation are required before operational use.
