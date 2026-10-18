# TrustXRay — Uncertainty-aware pneumonia detection on chest X-rays

> An AI that detects pneumonia on pediatric chest X-rays, **tells how confident it is**, **shows where it looked**, **refers uncertain cases to a radiologist**, and **refuses images it was not trained for**.

### 🔗 [**Try the live demo → trustxray.streamlit.app**](https://trustxray.streamlit.app/)

> ⚠️ **Research demo — not a medical device.** Not clinically validated. Do not use for diagnosis.

![App screenshot](readme_figures/app_screenshot.png)

---

## Why this project
Most medical AI models answer with the same confidence whether they are right or wrong. A clinician cannot trust a tool that never says *"I don't know"*. TrustXRay combines:

| Component | Method | Purpose |
|---|---|---|
| Classification | ResNet18, transfer learning (ImageNet) | Detect pneumonia |
| Uncertainty | Monte Carlo Dropout (30 passes) | Know when the model is unsure |
| Referral | Uncertainty cut-off fixed on validation | Send doubtful cases to a radiologist |
| Explainability | Grad-CAM | Check that the model looks at the lungs |
| Out-of-distribution detection | Mahalanobis distance on ResNet18 features | Refuse images that are not chest X-rays |
| Demo | Streamlit | Let anyone test the model |

## Key results (test set, 624 X-rays never seen during training)

| | All cases | Cases kept by the AI (after referral) |
|---|---|---|
| AUC | **0.990** | — |
| Sensitivity (pneumonia detected) | 99.5% | **100%** |
| Missed pneumonia (FN) | 2 | **0** |
| False alarms (FP) | 29 | 16 |
| Accuracy | 95.0% | 96.9% |

**The uncertainty-based referral sends 16.8% of X-rays to a radiologist and catches 2 / 2 missed pneumonia cases**: on the cases it keeps, the AI sends no sick child home by mistake.

## Method and main findings

### 1. Data exploration — [`01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)
[PneumoniaMNIST](https://medmnist.com/) (MedMNIST v2, 224×224): 5,856 pediatric chest X-rays (Kermany et al., 2018), CC BY 4.0.

![Class distribution](readme_figures/01_class_distribution.png)

- **Class imbalance** (~2.9 pneumonia : 1 normal) → class-weighted loss.
- **Distribution shift**: 74.2% pneumonia in train vs 62.5% in test → threshold tuned on validation only; evaluation with sensitivity & specificity.
- **Accuracy is misleading**: always predicting "pneumonia" already gives 62.5% on test.
- **Possible shortcut**: some images contain a laterality marker ("R") → checked later with Grad-CAM.

### 2. Baseline model — [`02_baseline_training.ipynb`](notebooks/02_baseline_training.ipynb)
ResNet18 pretrained on ImageNet, Dropout(0.3) head, class-weighted BCE loss, augmentation **without horizontal flip** (the body is not symmetric: the heart is on the left).

| Confusion matrix (threshold tuned on validation) | ROC curves |
|---|---|
| ![Confusion matrix](readme_figures/02_cm_test_tuned.png) | ![ROC](readme_figures/02_roc_curves.png) |

- The tuned threshold halves missed pneumonia (7 → 3) at the cost of more false alarms (16 → 29): **choosing the operating point is a clinical decision** (screening favors sensitivity).
- Validation (AUC 0.999) is more optimistic than test (AUC 0.990): the distribution shift predicted in step 1 is confirmed.
- Probabilities are **not calibrated** (class weighting shifts them; threshold = 0.043) → the app shows votes and uncertainty, not a misleading percentage.

### 3. Uncertainty — [`03_uncertainty_mc_dropout.ipynb`](notebooks/03_uncertainty_mc_dropout.ipynb)
Each X-ray is analysed 30 times with dropout active. Uncertainty score **z** = distance between the mean logit and the decision threshold, in standard deviations.

| Errors are more uncertain | Referral curve |
|---|---|
| ![Uncertainty](readme_figures/03_uncertainty_correct_vs_errors.png) | ![Referral](readme_figures/03_referral_curve.png) |

- Median z: 23.0 (correct) vs 13.3 (errors); AUROC of uncertainty for detecting errors = 0.795.
- The referral rule (fixed on validation) catches **2 / 2 missed pneumonia** cases.
- Limits: the rule refers 10% of validation cases but 16.8% of test cases (distribution shift again), and some errors remain *confident* — uncertainty is a safety net, not a guarantee.

### 4. Explainability — [`04_gradcam_explainability.ipynb`](notebooks/04_gradcam_explainability.ipynb)

![Grad-CAM on pneumonia](readme_figures/04_gradcam_true_positive.png)

- On detected pneumonia, the heat falls on the **lung fields**.
- **Shortcut check**: for X-rays predicted as pneumonia, only 30.2% of the heat is in the image border (uniform map = 41%) and only 3 / 416 focus on the border → **no evidence that the model uses the "R" marker**.
- **Confident false alarms** (below): the model looks inside the lungs, on normal X-rays that look whiter / hazier. This is a visual hypothesis only: simply changing the brightness or contrast of a normal X-ray did **not** flip the prediction (see robustness tests), so the cause of these errors remains open.

![Grad-CAM on false positives](readme_figures/04_gradcam_false_positive.png)

### 5. Out-of-distribution detection — [`05_ood_detection.ipynb`](notebooks/05_ood_detection.ipynb)
**Problem found while testing the app:** a blank image, random noise or a grayscale text document were classified as *pneumonia* with **30 / 30 votes**. MC Dropout measures doubt *on chest X-rays*; it cannot tell that an image is **not** a chest X-ray.

**Solution:** Mahalanobis distance (Lee et al., 2018) between the 512 ResNet18 features of an image and those of the training X-rays (Ledoit-Wolf covariance). Limit = 99th percentile of validation distances.

![OOD distances](readme_figures/05_ood_distances.png)

| Test set | Rejected | AUROC vs real X-rays |
|---|---|---|
| **Real test X-rays** | **1.1%** (98.9% accepted) | — |
| Blank / uniform images | 100% | 1.000 |
| Random noise | 100% | 1.000 |
| Text documents | 100% | 1.000 |
| X-rays rotated 90° | 99.0% | 0.999 |
| Breast ultrasound (BreastMNIST) | 99.4% | 1.000 |
| Natural photos (CIFAR-10) | 97.5% | 0.999 |
| Inverted X-rays (negative) | 41.0% | 0.975 |

- In the app, the decision has **3 zones**: distance ≤ limit → the AI decides; slightly above (≤ 1.2 × limit) → *unusual image, referred to a radiologist*; far above → **refused**. A real X-ray just over the limit is therefore referred, not rejected.
- Limit: inverted X-rays keep the same anatomy and often pass.

### 6. Demo app and end-to-end tests — [`app/`](app/)
Streamlit app deployed on Streamlit Community Cloud; the trained model is published as a [GitHub release](https://github.com/Adib008/TrustXRay/releases/tag/v1.0). Pipeline: color / blank image filter → OOD check → MC Dropout (30 passes) → referral rule → Grad-CAM.

The whole app was tested end to end with an automated browser (6 example X-rays + 15 trap images, real model):

| Input | App answer |
|---|---|
| Real test X-rays, mirrored, brighter, 2000 px JPEG | Correct prediction (accepted) |
| Missed pneumonia #148 | **Uncertain → referred** (caught) |
| Darker X-ray | Uncertain → referred |
| Real X-ray just above the OOD limit (#35) | Unusual image → referred |
| Rotated 90°, strong noise, strong blur, very low JPEG quality, zoomed crop | **Refused** (out of distribution) |
| Text document | **Refused** (out of distribution) |
| Blank white / black image | **Refused** (almost uniform) |
| Color photo, sepia-tinted X-ray | **Refused** (not grayscale) |

A concurrency bug found online (Grad-CAM hook left on the shared cached model) was fixed with a lock and guaranteed hook removal.

## Limitations
- Pediatric patients only (1–5 years), single hospital (Guangzhou), low-resolution resized images.
- Not clinically validated, not a medical device.
- Monte Carlo Dropout is applied to the last layer only (a light approximation of Bayesian deep learning).
- Grad-CAM is coarse (7×7) and shows correlation, not causation; no lung masks were available.
- The OOD detector misses most inverted X-rays and was not tested on adult chest X-rays or other body parts.

## Future work
- **Calibration** (temperature scaling) to display reliable probabilities.
- **Full Bayesian uncertainty**: MC Dropout in all layers or deep ensembles.
- **External validation** on adult X-rays from other hospitals (e.g. ChestX-ray14, CheXpert).
- **Robustness training** (noise, contrast augmentation) and lung segmentation to make Grad-CAM checks quantitative.

## Project structure
```
notebooks/   01 exploration · 02 training · 03 uncertainty · 04 Grad-CAM · 05 OOD detection
app/         Streamlit demo (app.py, config, OOD statistics, example X-rays)
readme_figures/  figures used in this README
```

## Authors
- **Adib Ouyachchi** — data exploration, model training, uncertainty estimation (MC Dropout), Streamlit app
- **Kenza El Ouaghlidi** — Grad-CAM explainability, demo examples, model release, deployment and OOD detector integration

Both authors contributed to the analysis and the interpretation of the results.  
ESISA (École Supérieure d'Ingénierie en Sciences Appliquées), Fès, Morocco — 2026.

## References
- Kermany D. et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 2018.
- Yang J. et al. *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 2023.
- He K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
- Gal Y., Ghahramani Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML, 2016.
- Lee K. et al. *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS, 2018.
- Selvaraju R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV, 2017.
