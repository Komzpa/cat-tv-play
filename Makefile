STATE_ROOT ?= $(HOME)/.openclaw/state/cat-tv-learning
YOLO_EXPORT ?= $(STATE_ROOT)/exports/sher-yolo-seg
YOLO_BASE_MODEL ?=
YOLO_MODEL ?= $(STATE_ROOT)/models/sher-yolo-seg.pt
YOLO_RUN_ROOT ?= $(STATE_ROOT)/yolo-runs
YOLO_DEVICE ?=
YOLO_IMGSZ ?= 960
YOLO_EPOCHS ?= 80
YOLO_BATCH ?= 8

.PHONY: export-sher-yolo-seg validate-sher-yolo-seg train-sher-yolo-seg eval-sher-yolo-seg

export-sher-yolo-seg:
	python3 scripts/export_cat_projector_yolo_segmentation.py \
		--labels-root "$(STATE_ROOT)/label-review/labels" \
		--output "$(YOLO_EXPORT)" \
		--symlink

validate-sher-yolo-seg:
	python3 scripts/export_cat_projector_yolo_segmentation.py \
		--labels-root "$(STATE_ROOT)/label-review/labels" \
		--validate-only

train-sher-yolo-seg:
	test -n "$(YOLO_BASE_MODEL)"
	python3 scripts/cat_projector_yolo_segmentation.py train \
		--dataset "$(YOLO_EXPORT)/dataset.yaml" \
		--base-model "$(YOLO_BASE_MODEL)" \
		--out "$(YOLO_MODEL)" \
		--run-root "$(YOLO_RUN_ROOT)" \
		--epochs "$(YOLO_EPOCHS)" \
		--imgsz "$(YOLO_IMGSZ)" \
		--batch "$(YOLO_BATCH)" \
		$(if $(YOLO_DEVICE),--device "$(YOLO_DEVICE)",)

eval-sher-yolo-seg:
	python3 scripts/cat_projector_yolo_segmentation.py eval \
		--dataset "$(YOLO_EXPORT)/dataset.yaml" \
		--model "$(YOLO_MODEL)" \
		--out "$(STATE_ROOT)/evals/sher-yolo-seg-$$(date -u +%Y%m%dT%H%M%SZ)" \
		$(if $(YOLO_DEVICE),--device "$(YOLO_DEVICE)",)
