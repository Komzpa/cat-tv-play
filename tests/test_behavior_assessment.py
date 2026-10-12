from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import json,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=spec_from_file_location('local_behavior_assessment',ROOT/'custom_components/cat_tv_play/behavior_assessment.py');m=module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def recording(tmp_path):
 p=tmp_path/'recording';p.mkdir();(p/'manifest.json').write_text(json.dumps({'ha_session_id':'session','source_meow_event_id':'meow','created_at':'2026-10-12T05:45:52Z'}));(p/'verified.json').write_text('{}');(p/'chunk_0000.mp4').write_bytes(b'fixture');return p

def assess(p, *, records=None, display=True, render=True, **kwargs):
 records=records if records is not None else [{'reason':'source_subtracted_calibrated_high_jump','height_cm':90,'cat_probability':.95,'model_path':'existing-model','chunk':0}]
 return m.assess_recording(p,model={'path':'existing-model','sha256':'model-version'},score_records=lambda chunks:records,verify_display=lambda chunks,r:{'verified':display},confirm_render=lambda chunks,r:{'max_height_cm':90 if render else None,'candidate_frame_count':2,'canonical_alert_record':r if render else None,'detector_context':{'backend':'source_subtracted'},'canonical_detector_route':'source_subtracted','canonical_detector_probability':.95,'canonical_detector_model':'existing-model'},**kwargs)

def test_independent_confirmed_play_keeps_exact_source_and_version(tmp_path):
 p=recording(tmp_path);r=assess(p);assert r['outcome']=='active_play';assert r['source_meow_event_id']=='meow';assert r['session_id']=='session';assert r['inputs']['chunks'][str(p/'chunk_0000.mp4')]==m.file_sha256(p/'chunk_0000.mp4')

@pytest.mark.parametrize('options',[{'records':[]},{'display':False},{'render':False},{'records':[{'reason':'calibrated_high_jump','height_cm':100,'cat_probability':1,'model_path':'existing-model','chunk':0}]},{'records':[{'reason':'source_subtracted_calibrated_high_jump','height_cm':100,'cat_probability':float('nan'),'model_path':'existing-model','chunk':0}]}])
def test_missing_visibility_output_or_invalid_teacher_abstains(tmp_path,options):
 r=assess(recording(tmp_path),**options);assert r['status']=='abstain';assert r['outcome']=='unknown';assert r['confidence']==0

def test_teacher_failure_abstains_without_endpoint_disclosure(tmp_path):
 p=recording(tmp_path)
 def fail(chunks):raise RuntimeError('private endpoint')
 r=m.assess_recording(p,model={'sha256':'version'},score_records=fail,verify_display=None,confirm_render=None);assert r['reason']=='assessment_error:RuntimeError';assert 'private' not in json.dumps(r)

def test_cache_keeps_prior_observed_play_when_later_window_has_no_jump(tmp_path):
 p=recording(tmp_path);r=assess(p);(p/'chunk_0001.mp4').write_bytes(b'later resting frame');assert assess(p,records=[])['assessment_id']==r['assessment_id']

def test_camera_model_update_invalidates_old_guess(tmp_path):
 p=recording(tmp_path);assess(p);r=m.assess_recording(p,model={'path':'existing-model','sha256':'new-version'},score_records=lambda chunks:[],verify_display=None,confirm_render=None);assert r['outcome']=='unknown';assert r['model']['sha256']=='new-version'


def test_nominal_backend_cannot_confirm_fallback_detector(tmp_path):
 p=recording(tmp_path)
 candidate={'reason':'source_subtracted_calibrated_high_jump','height_cm':90,'cat_probability':.95,'model_path':'existing-model','chunk':0}
 r=m.assess_recording(p,model={'path':'existing-model','sha256':'version'},score_records=lambda chunks:[candidate],verify_display=lambda chunks,r:{'verified':True},confirm_render=lambda chunks,r:{'max_height_cm':90,'candidate_frame_count':2,'canonical_alert_record':r,'detector_context':{'backend':'source_subtracted'}})
 assert r['outcome']=='unknown'


def test_incremental_scan_does_not_skip_older_unseen_chunks(tmp_path):
 p=recording(tmp_path)
 for index in range(1,6):(p/f'chunk_{index:04d}.mp4').write_bytes(str(index).encode())
 first=assess(p,records=[],max_chunks=2)
 assert list(first['inputs']['chunks'])==[str(p/'chunk_0000.mp4'),str(p/'chunk_0001.mp4')]
 second=assess(p,records=[],max_chunks=2)
 assert list(second['inputs']['chunks'])==[str(p/'chunk_0002.mp4'),str(p/'chunk_0003.mp4')]
 assert len(second['coverage']['assessed_chunks'])==4

def test_explicit_history_selection_requires_original_bounded_chunks(tmp_path):
 p=recording(tmp_path)
 with pytest.raises(ValueError):assess(p,selected_chunks=[tmp_path/'outside.mp4'])
 with pytest.raises(ValueError):assess(p,max_chunks=5)


def test_changed_manifest_invalidates_cached_source_link(tmp_path):
 p=recording(tmp_path);assess(p,records=[])
 manifest=json.loads((p/'manifest.json').read_text());manifest['source_meow_event_id']='different-event';(p/'manifest.json').write_text(json.dumps(manifest))
 assert assess(p,records=[])['source_meow_event_id']=='different-event'
