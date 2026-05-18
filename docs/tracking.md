# Wall-Plane Tracking

Jump review uses this chain:

```text
detection -> robust point -> wall coordinates -> outlier gate -> Kalman update -> peak extraction
```

False detections must be rejected before they update the tracker. The Kalman
state is only a continuity model, not a cat/not-cat classifier.

## Data Model

`custom_components/cat_tv_play/tracking.py` tracks detections after calibration:

- `t`: frame timestamp in seconds.
- `x_cm`, `y_cm`: wall-plane centimeters from the calibrated wall homography.
- `confidence`: detector or reviewer confidence.
- `area_px`: original candidate area, used only as a weak tie-breaker/noise hint.
- `source`: detector family, for review/debug output.

Do not track raw bbox corners in image pixels. Convert the selected point into
wall coordinates first, then apply physical gates in centimeters.

## Candidate Top Point

Source subtraction returns candidates, not final cat identities. Its connected
component top point is deliberately robust: it uses a percentile upper cap and a
median point, so one isolated high residual pixel cannot create a fake apex.

## Gates

The tracker rejects impossible candidates before update:

- below the floor or above the wall limit;
- horizontal or vertical teleport in wall centimeters per second;
- Mahalanobis distance too far from the predicted track.

Confidence changes measurement noise. It does not give a high-confidence false
positive permission to teleport.

## Peak Height

Peak extraction uses accepted raw wall heights, not the smoothed Kalman `y`.
Smoothing is useful for continuity and display, but it can push a real jump apex
down. Review output should keep both raw accepted height and filtered height.

The metric is wall-plane reach/contact height. If the point is not actually on
the wall plane, camera parallax can still affect the physical interpretation.
