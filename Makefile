STATE_ROOT ?= $(HOME)/.openclaw/state/cat-tv-learning
YOLO_EXPORT ?= $(STATE_ROOT)/exports/sher-yolo-seg
YOLO_BASE_MODEL ?= $(STATE_ROOT)/models/sher-yolo-seg-first.pt
YOLO_MODEL ?= $(STATE_ROOT)/models/sher-yolo-seg.pt
YOLO_RUN_ROOT ?= $(STATE_ROOT)/yolo-runs
YOLO_DEVICE ?=
YOLO_IMGSZ ?= 960
YOLO_EPOCHS ?= 80
YOLO_BATCH ?= 8
YOLO_SEED ?= 20260523
YOLO_OPTIMIZER ?= AdamW
YOLO_LR0 ?= 0.0002
YOLO_LRF ?= 0.1
YOLO_WARMUP_EPOCHS ?= 0
YOLO_FREEZE ?=
YOLO_CONFIDENCE_THRESHOLD ?= 0.55
YOLO_ALLOW_NEW_LINEAGE ?=
YOLO_TRAIN_OPTIONAL_ARGS := $(if $(YOLO_ALLOW_NEW_LINEAGE),--allow-new-sher-lineage,) $(if $(YOLO_DEVICE),--device "$(YOLO_DEVICE)",) $(if $(YOLO_FREEZE),--freeze "$(YOLO_FREEZE)",)
YOLO_EVAL_OPTIONAL_ARGS := $(if $(YOLO_DEVICE),--device "$(YOLO_DEVICE)",)

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
		--optimizer "$(YOLO_OPTIMIZER)" \
		--lr0 "$(YOLO_LR0)" \
		--lrf "$(YOLO_LRF)" \
		--warmup-epochs "$(YOLO_WARMUP_EPOCHS)" \
		$(YOLO_TRAIN_OPTIONAL_ARGS) --seed "$(YOLO_SEED)"

eval-sher-yolo-seg:
	python3 scripts/cat_projector_yolo_segmentation.py eval \
		--dataset "$(YOLO_EXPORT)/dataset.yaml" \
		--model "$(YOLO_MODEL)" \
		--out "$(STATE_ROOT)/evals/sher-yolo-seg-$$(date -u +%Y%m%dT%H%M%SZ)" \
		$(YOLO_EVAL_OPTIONAL_ARGS) --confidence-threshold "$(YOLO_CONFIDENCE_THRESHOLD)"
