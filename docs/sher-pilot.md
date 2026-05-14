# Sher Pilot Notes

These notes describe the pilot that motivated Cat TV Play. They are examples,
not required configuration.

The pilot used:

- a projector as the display media player;
- a projector-mounted camera looking at the wall;
- generated and downloaded Cat TV clips;
- session recordings for review;
- one-meter cardboard angle profiles for wall calibration.

Important lessons:

- The cat detector can miss a visible cat near the lower-left edge, so review
  video and human labels matter.
- The physical floor is not the bottom of the projected image.
- Perspective correction matters. A single `cm/px` ratio underestimates or
  overestimates jump height depending on where the cat is.
- Realistic bug-like motion triggered higher jumps than low mouse-hole motion.
- The best observed jump reached about 180 cm by forepaws after homography
  calibration.

The useful automation boundary is:

1. A private automation decides when the cat is asking to play.
2. Cat TV Play starts and records a generic session.
3. Review/camera tooling supplies observations and calibrated jump heights.
4. Private logic chooses the next clip.

