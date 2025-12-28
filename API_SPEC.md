# API Specification (Draft)

## Purpose
Define stable output contracts for integration with security and enterprise pipelines.

## Inference command
`media-auth-forensics infer --input <path> --out report.json`

## Output JSON (high-level)
- file: string
- sha256: string
- is_video: boolean
- frames_analyzed: integer
- final_score_max: float [0..1]
- final_score_mean: float [0..1]
- temporal_binary_pattern: array[int] (0/1)
- frames: array[FrameReport]

### FrameReport
- frame_index: int
- timestamp: float (seconds) or null
- region_score: float [0..1]
- worst_region_score: float [0..1]
- worst_variant: string
- adversarial_variants: array[{variant: string, score: float}]
- regions: array[{bbox: [x,y,w,h], area: int, mean_score: float}]
- face_predictions: array[{bbox: [x,y,w,h], face_score: float, conf: float}]
- model_identifier: {probs: array[float]} or null
- final_frame_score: float [0..1]

## Stability policy
- Fields will not be removed within a major version.
- New fields may be added in minor versions.
