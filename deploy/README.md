# Deployment

`private.example.yml` documents every site-specific value this project needs.
Copy it to `private.yml` beside it and fill it in; git ignores that copy.

The units in `systemd/` are user units. They read one environment file,
`~/.config/cat-tv-play/environment`, so nothing site-specific is baked into a
tracked unit. That file sets the same values as `private.yml` through the
environment variables listed in `cat_tv_play_config.ENV_OVERRIDES`, plus the
paths the units themselves use:

```
CAT_TV_PLAY_CONFIG=%h/.config/cat-tv-play/private.yml
CAT_TV_STATE_ROOT=/var/lib/cat-tv-play/learning
CAT_TV_RECORDINGS_ROOT=/var/lib/cat-tv-play/learning/recordings
CAT_TV_HA_URL=http://home-assistant.example.test:8123
CAT_TV_HA_TOKEN_FILE=%h/.config/cat-tv-play/ha-token
CAT_TV_CAMERA_SNAPSHOT_URL=http://camera.example.test:8081/shot.jpg
CAT_PROJECTOR_SAM_ENDPOINT=http://127.0.0.1:8766/segment
CAT_PROJECTOR_SAM_CHECKPOINT=/var/lib/cat-tv-play/learning/sam-models/sam_vit_b_01ec64.pth
```

Install with `systemctl --user link` from a checkout at
`~/.local/share/cat-tv-play/checkout`, or copy the units and adjust the paths.
A unit whose data root lives on a separate filesystem should also carry
`RequiresMountsFor=` and an `ExecStartPre=` marker test, so it fails closed
instead of writing into an empty mountpoint.
