# Calibration Notes

Cat TV Play measures in the wall plane. The camera image is only a source of
coordinates; the stored homography turns those image coordinates into wall
centimeters.

## Marker Layout

A good marker layout has:

- two vertical one-meter references;
- one horizontal one-meter reference;
- at least one known point touching the physical floor;
- all references in the same plane as the projected target.

Do not put the ruler on a chair, screen frame, or nearby furniture unless that
is also the plane where the cat touches the target.

## Coordinate System

Use centimeters:

- `wall_y_cm = 0` at the physical floor;
- `wall_y_cm = 100` at one meter above the floor;
- `wall_x_cm = 0` wherever convenient, usually under a central vertical marker;
- left of origin can be negative.

The bottom of the projected image is not necessarily the physical floor. Always
measure jumps from wall coordinates, not from video source pixels.

## Picking Points

For each marker endpoint, record:

- `image_x`
- `image_y`
- `wall_x_cm`
- `wall_y_cm`

Use six points when possible, even though four are enough mathematically. More
points reduce the impact of hand-picking error.

## Measuring A Jump

Pick the highest forepaw point in the frame. If motion blur makes the paw hard
to locate, record a range in your note and keep the exact clicked point as the
machine-readable measurement.

```yaml
service: cat_tv_play.measure_image_point
data:
  calibration_id: living_room_wall
  image_x: 253
  image_y: 220
response_variable: paw
```

Then save the observation:

```yaml
service: cat_tv_play.record_observation
data:
  behavior: jump
  image_x: 253
  image_y: 220
  wall_x_cm: "{{ paw.wall_x_cm }}"
  wall_y_cm: "{{ paw.wall_y_cm }}"
  jump_height_cm: "{{ paw.jump_height_cm }}"
```

