"""Constants for Cat TV Play."""

DOMAIN = "cat_tv_play"

CONF_MEDIA_PLAYER_ENTITY_ID = "media_player_entity_id"
CONF_CAMERA_ENTITY_ID = "camera_entity_id"
CONF_DEFAULT_MEDIA_URL = "default_media_url"
CONF_RECORDER_SWITCH_ENTITY_IDS = "recorder_switch_entity_ids"
CONF_SNAPSHOT_SWITCH_ENTITY_IDS = "snapshot_switch_entity_ids"

SERVICE_START_SESSION = "start_session"
SERVICE_STOP_SESSION = "stop_session"
SERVICE_RECORD_OBSERVATION = "record_observation"
SERVICE_SAVE_CALIBRATION = "save_calibration"
SERVICE_MEASURE_IMAGE_POINT = "measure_image_point"

EVENT_SESSION_STARTED = f"{DOMAIN}_session_started"
EVENT_SESSION_STOPPED = f"{DOMAIN}_session_stopped"
EVENT_OBSERVATION_RECORDED = f"{DOMAIN}_observation_recorded"
EVENT_CALIBRATION_SAVED = f"{DOMAIN}_calibration_saved"

STORAGE_VERSION = 1
STORAGE_KEY = DOMAIN
