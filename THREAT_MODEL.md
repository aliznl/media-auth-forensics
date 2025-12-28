# Threat Model

## Objective
Identify and mitigate threats against a media manipulation detection system.

---

## Adversary Capabilities

Assumed attacker abilities:
- Access to modern generative models
- Ability to post-process media
- Knowledge of detection techniques
- Ability to perform repeated attempts

---

## Threat Categories

### 1. Media-Level Attacks
- JPEG recompression
- Blur and denoising
- Resolution changes
- Cropping and scaling

### 2. Spectral Attacks
- Frequency smoothing
- GAN fingerprint removal
- Low-pass filtering

### 3. Model Attacks
- Overfitting exploitation
- Transfer attacks
- Black-box probing

### 4. System Attacks
- Malformed file uploads
- Resource exhaustion
- API abuse

---

## Mitigations

- Ensemble detection (ML + forensic)
- Adversarial evaluation
- Input validation and limits
- Rate limiting (API deployments)
- Human review escalation

---

## Residual Risk

No detection system is perfect.
Residual risk must be managed through:
- Continuous retraining
- Monitoring
- Policy enforcement

---

End of document
