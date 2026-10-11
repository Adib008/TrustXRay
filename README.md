# TrustXRay — Uncertainty-aware pneumonia detection on chest X-rays

> An AI that detects pneumonia on pediatric chest X-rays, **tells how confident it is**, **shows where it looked**, and **refers uncertain cases to a radiologist**.

🚧 *Work in progress — built in one week as a learning project in AI for healthcare.*

## Why this project
Most medical AI models answer with the same confidence whether they are right or wrong. A clinician cannot trust a tool that never says *"I don't know"*. TrustXRay combines:

- **Deep learning** (transfer learning on a pretrained CNN) for classification
- **Uncertainty estimation** (Monte Carlo Dropout) to flag doubtful cases
- **Explainability** (Grad-CAM) to check that the model looks at the lungs
- **An interactive demo** (Streamlit)

## Dataset
[PneumoniaMNIST](https://medmnist.com/) (MedMNIST v2, 224×224), derived from 5,856 pediatric chest X-rays (Kermany et al., *Cell*, 2018). License: CC BY 4.0.

| Split | Normal | Pneumonia | % pneumonia |
|---|---|---|---|
| Train | 1,214 | 3,494 | 74.2% |
| Val | 135 | 389 | 74.2% |
| Test | 234 | 390 | 62.5% |

**Key findings from exploration**
- Strong class imbalance (~3:1) → class-weighted loss.
- Distribution shift between train (74.2%) and test (62.5%) → decision threshold tuned on validation only; evaluation with sensitivity & specificity (prevalence-independent).
- Accuracy is misleading: always predicting "pneumonia" already gives 62.5% on test.
- Some images contain a laterality marker ("R") → possible shortcut, to be checked with Grad-CAM.

## Project structure
```
notebooks/   01_data_exploration.ipynb, 02_training.ipynb, ...
app/         Streamlit demo
figures/     generated plots
```

## Progress
- [x] 01 — Data exploration
- [ ] 02 — Baseline model (transfer learning)
- [ ] 03 — Uncertainty (Monte Carlo Dropout)
- [ ] 04 — Explainability (Grad-CAM)
- [ ] 05 — Evaluation
- [ ] 06 — Streamlit demo

## Limitations
Pediatric patients only (1–5 years), single hospital, resized images. **Not a medical device and not clinically validated.**

## References
- Kermany D. et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 2018.
- Yang J. et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 2023.
- Gal Y., Ghahramani Z. *Dropout as a Bayesian Approximation.* ICML, 2016.
- Selvaraju R. et al. *Grad-CAM: Visual Explanations from Deep Networks.* ICCV, 2017.
