"""Frozen four-recording, prefix-causal temporal-sketch pilot runner."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Spatial.algorithms.adaptive_range_search import AdaptiveRangeSearch
from Spatial.algorithms.causal_temporal_sketch import CausalTemporalSketch, PRE_ALIGNMENT_SAMPLES
from Spatial.algorithms.causal_temporal_baselines import CausalTemporalBaselines, legacy_peak_normalize
from Spatial.algorithms.detection import bandpass_filter,get_peak_amplitudes
from Spatial.algorithms.spatial_footprint import build_knn_table_with_self,com_features,extract_local_p2p,footprint_p2p_features
from Spatial.data.loader import Dataset, list_hybrid_janelia_scenes, load_hybrid_janelia, load_mearec_npz
from Spatial.experiments.run_adaptive_5bit_range_search import labels_for_peak_output
from Spatial.experiments.run_mearec_method_sweep import DEFAULT_NPZ_DIR, discover_npz
from Spatial.experiments.causal_temporal_selection import (
 BASE_REPOSITORY_REVISION, DELAYS, FEATURE_COUNTS, IMPLEMENTATION_SOURCE_PATHS, LOCKED_INPUT_SUBSET_SHA256, LOCKED_PILOT_SOURCE_SHA256, PILOT_DATASET_IDS, TEMPORAL_INPUT_CONTRACT, TEMPORAL_INPUT_CONTRACT_SHA256, build_selection_manifest, validate_development_rows,
 write_json_deterministic,
)

BITS=5; DESCRIPTOR_DIM=9; WINDOW=64; POST_ALIGNMENT_SAMPLES=48
FILTER_LOW_HZ=300.0; FILTER_HIGH_HZ=6000.0; FILTER_ORDER=3
@dataclass(frozen=True)
class PilotRecordSpec:
 dataset_id: str
 family: str
 source_input_path: str
 source_file_sha256: str
 source_input_subset_digest: str
 load_dataset: Callable[[], Dataset]

def sha256_json(value:Any)->str: return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _digest_rows(event_ids,times,labels,central): return sha256_json({'event_row_id':np.asarray(event_ids).tolist(),'time':np.asarray(times).tolist(),'label':np.asarray(labels).tolist(),'central':np.asarray(central).tolist()})

def implementation_source_snapshot() -> dict[str,Any]:
 """Content-address the exact local source consumed by this pilot invocation."""
 root=Path(__file__).resolve().parents[1]
 try:
  observed=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
 except Exception as error:
  raise ValueError('cannot verify repository base revision') from error
 if observed!=BASE_REPOSITORY_REVISION: raise ValueError('repository HEAD does not match the frozen base revision')
 files={relative:_sha256_file(root/relative) for relative in IMPLEMENTATION_SOURCE_PATHS}
 return {'repository_base_revision':BASE_REPOSITORY_REVISION,'implementation_source_sha256':files,'implementation_source_digest':sha256_json(files)}

def validate_prepared_temporal_provenance(prepared: Mapping[str,Any]) -> None:
 """Fail closed unless prepared rows state the frozen raw/legacy temporal route."""
 provenance=prepared.get('provenance')
 if not isinstance(provenance,Mapping): raise ValueError('prepared recording is missing temporal provenance')
 expected_filter={'kind':'Butterworth bandpass','low_cutoff_hz':FILTER_LOW_HZ,'high_cutoff_hz':FILTER_HIGH_HZ,'order':FILTER_ORDER,'zero_phase':True,'implementation':'scipy.signal.filtfilt(axis=1)'}
 expected_window={'pre_alignment_samples':PRE_ALIGNMENT_SAMPLES,'post_alignment_samples':POST_ALIGNMENT_SAMPLES,'interval':'[t-15, t+49)','samples':WINDOW}
 n_events=provenance.get('n_events')
 if isinstance(n_events,bool) or not isinstance(n_events,(int,np.integer)) or int(n_events)!=len(np.asarray(prepared.get('times',[]))) or provenance.get('recording_input')!='dataset.raw_data (channel x sample)' or provenance.get('sampling_frequency_hz')!=30000 or provenance.get('temporal_input')!='filtered_pre_event_normalization_main_channel' or provenance.get('filter')!=expected_filter or provenance.get('alignment_source')!='get_peak_amplitudes(filtered, dataset.spike_times, window=PRE_ALIGNMENT_SAMPLES) preserves boundary-filtered input order' or provenance.get('central_channel_source')!='get_peak_amplitudes: np.argmax(abs(per-channel signed peak amplitude)); exact ties select the smallest channel index' or provenance.get('central_post_alignment_lookahead_samples')!=14 or provenance.get('window')!=expected_window or provenance.get('raw_waveform_source')!='filtered central channel' or provenance.get('legacy_waveform_source')!='per-row max-absolute normalized raw_waveforms':
  raise ValueError('prepared temporal provenance does not match the frozen pilot contract')
def stable_three_way_split(times):
 t=np.asarray(times,dtype=np.int64); b=np.flatnonzero(t[1:]!=t[:-1])+1
 if t.size<3 or b.size<2 or np.any(t[1:]<t[:-1]): raise ValueError('no legal stable three-way split')
 best=None; target=.75*t.size
 for i,a in enumerate(b[:-1]):
  later=b[i+1:]; j=int(np.searchsorted(later,target))
  for q in (j-1,j):
   if 0<=q<later.size:
    c=int(later[q]); z=((a-.5*t.size)**2+(c-.75*t.size)**2,int(a),c)
    if best is None or z<best: best=z
 _,a,c=best; return np.arange(a),np.arange(a,c),np.arange(c,t.size)

def prepare_causal_events(dataset:Dataset,k_neighbors:int=7)->dict[str,np.ndarray|dict[str,Any]]:
 """One filter pass, joint masks, raw/legacy waveform rows, stable identity."""
 filtered=bandpass_filter(dataset.raw_data,dataset.fs,FILTER_LOW_HZ,FILTER_HIGH_HZ,FILTER_ORDER)
 _,times,central=get_peak_amplitudes(filtered,dataset.spike_times,window=PRE_ALIGNMENT_SAMPLES)
 labels=labels_for_peak_output(dataset.spike_times,dataset.spike_units,times,dataset.n_samples,window=PRE_ALIGNMENT_SAMPLES)
 original=np.flatnonzero((dataset.spike_times>=PRE_ALIGNMENT_SAMPLES)&(dataset.spike_times<dataset.n_samples-PRE_ALIGNMENT_SAMPLES))
 joint=times+POST_ALIGNMENT_SAMPLES<dataset.n_samples
 times,central,labels,original=times[joint],central[joint],labels[joint],original[joint]
 if times.size==0: raise ValueError('no events remain after causal 64-sample boundary filtering')
 table=build_knn_table_with_self(dataset.geom,min(k_neighbors,dataset.n_channels)); p2p,returned,neighbors=extract_local_p2p(filtered,times,central,table,window=PRE_ALIGNMENT_SAMPLES)
 assert np.array_equal(returned,times) and len(times)==len(labels)==len(central)==len(original)
 descriptor=np.column_stack([com_features(p2p,neighbors,dataset.geom),footprint_p2p_features(p2p,normalize=True)])
 raw=np.stack([filtered[int(ch),int(t)-PRE_ALIGNMENT_SAMPLES:int(t)+POST_ALIGNMENT_SAMPLES+1] for t,ch in zip(times,central)])
 if raw.shape!=(times.size,WINDOW) or descriptor.shape!=(times.size,DESCRIPTOR_DIM): raise ValueError('causal preparation did not produce N x 64 and N x D9 rows')
 if not np.all(np.isfinite(descriptor)) or not np.all(np.isfinite(raw)): raise ValueError('causal preparation produced non-finite rows')
 legacy=legacy_peak_normalize(raw); order=np.argsort(times,kind='stable')
 event_ids,times,labels,central,descriptor,raw,legacy=(x[order] for x in (original,times,labels,central,descriptor,raw,legacy))
 digest=_digest_rows(event_ids,times,labels,central)
 return {'event_row_id':event_ids,'times':times,'labels':labels,'central':central,'descriptor':descriptor,'raw_waveforms':raw,'legacy_normalized_waveforms':legacy,'provenance':{'row_digest':digest,'n_events':int(times.size),'recording_input':'dataset.raw_data (channel x sample)','sampling_frequency_hz':int(dataset.fs),'temporal_input':'filtered_pre_event_normalization_main_channel','filter':{'kind':'Butterworth bandpass','low_cutoff_hz':FILTER_LOW_HZ,'high_cutoff_hz':FILTER_HIGH_HZ,'order':FILTER_ORDER,'zero_phase':True,'implementation':'scipy.signal.filtfilt(axis=1)'},'alignment_source':'get_peak_amplitudes(filtered, dataset.spike_times, window=PRE_ALIGNMENT_SAMPLES) preserves boundary-filtered input order','central_channel_source':'get_peak_amplitudes: np.argmax(abs(per-channel signed peak amplitude)); exact ties select the smallest channel index','central_post_alignment_lookahead_samples':14,'window':{'pre_alignment_samples':PRE_ALIGNMENT_SAMPLES,'post_alignment_samples':POST_ALIGNMENT_SAMPLES,'interval':'[t-15, t+49)','samples':WINDOW},'raw_waveform_source':'filtered central channel','legacy_waveform_source':'per-row max-absolute normalized raw_waveforms'}}

def frozen_uniform_candidate_source(x_fit,y_fit,x_cal,y_cal,x_test):
 """Direct AdaptiveRangeSearch uniform D9/B5, calibration-only p99.9 radii."""
 x_fit=np.asarray(x_fit,dtype=float); x_cal=np.asarray(x_cal,dtype=float); x_test=np.asarray(x_test,dtype=float); y_fit=np.asarray(y_fit); y_cal=np.asarray(y_cal)
 for name,x in (('x_fit',x_fit),('x_cal',x_cal),('x_test',x_test)):
  if x.ndim!=2 or x.shape[1]!=DESCRIPTOR_DIM or not np.all(np.isfinite(x)): raise ValueError(f'{name} must be finite N x D9')
 if x_fit.shape[0]==0 or y_fit.ndim!=1 or y_fit.shape[0]!=x_fit.shape[0]: raise ValueError('x_fit and y_fit must contain aligned nonempty rows')
 if y_cal.ndim!=1 or y_cal.shape[0]!=x_cal.shape[0]: raise ValueError('x_cal and y_cal must contain aligned rows')
 model=AdaptiveRangeSearch(n_bits=5,fixed_unit_interval=True,radius_percentile=99.9).fit(x_fit,y_fit)
 codes_cal=model.transform(x_cal); index={u.item() if isinstance(u,np.generic) else u:i for i,u in enumerate(model.units_)}
 radii=model.radii_.copy(); calibration_counts=np.zeros(len(model.units_),dtype=np.int64); calibration_fallback_mask=np.zeros(len(model.units_),dtype=bool); radius_source=np.empty(len(model.units_),dtype=object)
 for unit,row in index.items():
  local=codes_cal[np.asarray(y_cal)==unit]
  calibration_counts[row]=local.shape[0]
  if local.size:
   radii[row]=int(np.ceil(np.percentile(np.abs(local-model.centroids_[row]).sum(axis=1),99.9,method='linear'))); radius_source[row]='calibration_p99.9_l1'
  else:
   calibration_fallback_mask[row]=True; radius_source[row]='no_calibration_rows_fit_p99.9'
 model.radii_=radii
 all_rows=np.ones((len(x_test),len(model.units_)),dtype=bool)
 codes_test=model.transform(x_test)
 distances=np.abs(codes_test[:,None,:]-model.centroids_[None,:,:]).sum(axis=2)
 matches=distances<=radii[None,:]
 candidates=[model.units_[row].copy() for row in matches]
 candidate_count=matches.sum(axis=1).astype(np.int64)
 row_comparisons=np.full(len(x_test),len(model.units_),dtype=np.int64)
 diag={'initial_candidate_count':candidate_count.copy(),'final_candidate_count':candidate_count.copy(),'widen_level':np.zeros(len(x_test),dtype=np.int64),'fallback':np.zeros(len(x_test),dtype=bool),'active_row_count':np.full(len(x_test),len(model.units_),dtype=np.int64),'row_comparisons':row_comparisons,'fallback_row_comparisons':np.zeros(len(x_test),dtype=np.int64),'total_row_comparisons_including_fallback':row_comparisons.copy()}
 level1_argmin_predictions=model.units_[np.argmin(distances,axis=1)].copy()
 payload54={'centroid_codes':model.centroids_.copy(),'radii':radii.copy(),'centroid_bits':BITS*DESCRIPTOR_DIM,'radius_capacity_bits':model.radius_capacity_bits,'bits_per_row':model.l1_row_bits_full_scale}
 return {'model':model,'candidates':candidates,'codes':codes_test,'diagnostics':diag,'calibration_radius_percentile':99.9,'calibration_counts':calibration_counts,'calibration_fallback_mask':calibration_fallback_mask,'radius_source':radius_source,'radii':radii.copy(),'all_rows':all_rows,'payload54':payload54,'level1_argmin_predictions':level1_argmin_predictions,'level1_source_scan_events':int(len(x_test)),'level1_source_unit_count':int(len(model.units_)),'level1_source_descriptor_quantizations':int(len(x_test)*DESCRIPTOR_DIM),'level1_source_centroid_reads':int(len(x_test)*len(model.units_)),'level1_source_l1_absolute_differences':int(len(x_test)*len(model.units_)*DESCRIPTOR_DIM),'level1_source_l1_reduction_additions':int(len(x_test)*len(model.units_)*(DESCRIPTOR_DIM-1)),'level1_source_radius_threshold_comparisons':int(len(x_test)*len(model.units_)),'level1_source_argmin_reduction_comparisons':int(len(x_test)*max(len(model.units_)-1,0))}

def distinct_unordered_pairs(candidates):
 return {tuple(sorted((a,b),key=repr)) for row in candidates for values in [list(dict.fromkeys(np.asarray(row).tolist()))] for i,a in enumerate(values) for b in values[i+1:]}
def candidate_support(candidates,labels):
 c=np.asarray([len(x) for x in candidates]); hit=np.asarray([y in x for y,x in zip(labels,candidates)])
 return {'candidate_recall':float(hit.mean()),'c0_events':int((c==0).sum()),'c1_events':int((c==1).sum()),'ambiguous_events':int((c>1).sum()),'distinct_unordered_pairs':len(distinct_unordered_pairs(candidates))}

def assert_future_tail_invariance(sketch:CausalTemporalSketch,waveforms,candidates):
 """Perturb future tail and prove core codes/features/SAD/predictions stay fixed."""
 wave=np.asarray(waveforms,float); altered=wave.copy(); altered[:,sketch.prefix_length:]+=17.25
 before_codes=sketch.quantize_prefix(wave); after_codes=sketch.quantize_prefix(altered); before_features=sketch.transform(wave); after_features=sketch.transform(altered); before_pred,before_diag=sketch.assign_candidates(wave,candidates); after_pred,after_diag=sketch.assign_candidates(altered,candidates)
 keys=('winning_sad_distance','logical_candidate_template_reads','logical_sad_absolute_differences','logical_sad_reduction_additions')
 return bool(np.array_equal(before_codes,after_codes) and np.array_equal(before_features,after_features) and np.array_equal(before_pred,after_pred) and sketch.sample_saturation_fraction(wave)==sketch.sample_saturation_fraction(altered) and all(np.array_equal(before_diag[k],after_diag[k]) for k in keys))


def _json_value(value: Any) -> Any:
 """Convert NumPy-bearing result structures without permitting NaN JSON."""
 if isinstance(value, Mapping): return {str(k):_json_value(v) for k,v in value.items()}
 if isinstance(value, (list,tuple)): return [_json_value(v) for v in value]
 if isinstance(value,np.ndarray): return _json_value(value.tolist())
 if isinstance(value,np.generic): return _json_value(value.item())
 if isinstance(value,float) and not np.isfinite(value): raise ValueError('result payload contains a non-finite value')
 return value


def _event_digest(prepared: Mapping[str,Any], indices: np.ndarray) -> str:
 idx=np.asarray(indices,dtype=np.int64)
 return _digest_rows(np.asarray(prepared['event_row_id'])[idx],np.asarray(prepared['times'])[idx],np.asarray(prepared['labels'])[idx],np.asarray(prepared['central'])[idx])


def _sha256_file(path: Path) -> str:
 digest=hashlib.sha256()
 with path.open('rb') as handle:
  for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
 return digest.hexdigest()


def _prediction_digest(predictions: Mapping[str,np.ndarray]) -> dict[str,str]:
 return {name:sha256_json(_json_value(np.asarray(values,dtype=object).tolist())) for name,values in predictions.items()}


def _prediction_metrics(predictions: np.ndarray, labels: np.ndarray, candidates: Sequence[np.ndarray]) -> dict[str,Any]:
 pred=np.asarray(predictions,dtype=object); truth=np.asarray(labels); ambiguous=np.asarray([len(row)>1 for row in candidates],dtype=bool)
 correct=np.asarray([p==y for p,y in zip(pred,truth)],dtype=bool); per_unit={}
 for unit in np.unique(truth):
  mask=truth==unit; per_unit[str(unit)]={'events':int(mask.sum()),'correct':int(correct[mask].sum()),'accuracy':float(correct[mask].mean())}
 return {'overall_correct':int(correct.sum()),'overall_accuracy':float(correct.mean()),'ambiguous_correct':int(correct[ambiguous].sum()),'ambiguous_accuracy':float(correct[ambiguous].mean()) if ambiguous.any() else 0.0,'per_unit':per_unit,'worst_unit_accuracy':min(v['accuracy'] for v in per_unit.values())}


def _with_spatial_fallback(predictions: np.ndarray,candidates: Sequence[np.ndarray],source: Mapping[str,Any]) -> np.ndarray:
 out=np.asarray(predictions,dtype=object).copy()
 c0=np.asarray([len(row)==0 for row in candidates],dtype=bool)
 out[c0]=np.asarray(source['level1_argmin_predictions'],dtype=object)[c0]
 return out


def _fisher_audit(sketch: CausalTemporalSketch) -> dict[str,Any]:
 ids=np.asarray(sketch.eligible_filter_ids_,dtype=np.int64)
 units=np.asarray(sketch.units_)
 return {'eligible_filter_ids':ids.tolist(),'between':np.asarray(sketch.fisher_between_sum_squares_,float).tolist(),'within':np.asarray(sketch.fisher_within_sum_squares_,float).tolist(),'scores':np.asarray(sketch.eligible_fisher_scores_,float).tolist(),'ranks':np.asarray(sketch.fisher_ranks_,dtype=np.int64)[ids].tolist(),'fit_unit_counts':{str(u):int(c) for u,c in zip(units,sketch.fit_unit_counts_)},'eligible_unit_mask':{str(u):bool(m) for u,m in zip(units,sketch.fisher_eligible_unit_mask_)},'eligible_unit_ids':_json_value(sketch.fisher_eligible_unit_ids_)}


def _baseline_columns(baselines: CausalTemporalBaselines,waveforms: np.ndarray,candidates: Sequence[np.ndarray],labels: np.ndarray,source: Mapping[str,Any]) -> tuple[dict[str,Any],dict[str,np.ndarray]]:
 """Evaluate all reports once on the frozen candidate lists and common C0 fallback."""
 mapping={'fit_scale_float64':'full_teacher','legacy_peak_normalized_float64':'legacy_full64','signed_full64x5':'signed_full64x5','fisher48_signed5':'fisher48_signed5','morphology12_unsigned5':'morphology12_unsigned5'}
 row:dict[str,Any]={}; predictions:dict[str,np.ndarray]={}
 for representation,prefix in mapping.items():
  initial,diagnostics=baselines.assign(waveforms,candidates,representation)
  final=_with_spatial_fallback(initial,candidates,source); predictions[prefix]=final; metric=_prediction_metrics(final,labels,candidates)
  for key,value in metric.items(): row[f'{prefix}_{key}']=value
  row[f'{prefix}_template_reads']=int(np.asarray(diagnostics['logical_candidate_template_reads']).sum())
  row[f'{prefix}_template_bits']=int(np.asarray(diagnostics['logical_candidate_template_bits']).sum())
  row[f'{prefix}_logical_sad_absolute_differences']=int(np.asarray(diagnostics['logical_sad_absolute_differences']).sum())
  row[f'{prefix}_logical_sad_reduction_additions']=int(np.asarray(diagnostics['logical_sad_reduction_additions']).sum())
  row[f'{prefix}_payload_accounting']=baselines.payload_accounting(representation)
 return row,predictions


def run_prepared_recording(prepared: Mapping[str,Any], *, record: PilotRecordSpec, implementation_snapshot: Mapping[str,Any] | None=None) -> list[dict[str,Any]]:
 """Run the frozen 15-point temporal grid from one already-prepared recording.

 The source, split, raw 64-window baseline fit, and every baseline prediction
 are intentionally constructed once; only prefix sketch fitting varies by grid
 configuration.  This is deliberately no-I/O and is suitable for synthetic tests.
 """
 if record.family not in PILOT_DATASET_IDS or record.dataset_id not in PILOT_DATASET_IDS[record.family]:
  raise ValueError('family must be hj or mearec')
 if not record.source_input_path or record.source_file_sha256.lower()!=LOCKED_PILOT_SOURCE_SHA256[record.dataset_id] or record.source_input_subset_digest!=LOCKED_INPUT_SUBSET_SHA256:
  raise ValueError('record source metadata does not match the locked pilot record specification')
 validate_prepared_temporal_provenance(prepared)
 snapshot=implementation_source_snapshot() if implementation_snapshot is None else dict(implementation_snapshot)
 if snapshot.get('repository_base_revision')!=BASE_REPOSITORY_REVISION or not isinstance(snapshot.get('implementation_source_sha256'),Mapping) or set(snapshot['implementation_source_sha256'])!=set(IMPLEMENTATION_SOURCE_PATHS) or snapshot.get('implementation_source_digest')!=sha256_json(dict(snapshot['implementation_source_sha256'])):
  raise ValueError('implementation source snapshot is malformed')
 names=('event_row_id','times','labels','central','descriptor','raw_waveforms','legacy_normalized_waveforms')
 arrays={key:np.asarray(prepared[key]) for key in names}
 n=len(arrays['times'])
 if n==0 or any(len(value)!=n for value in arrays.values()):
  raise ValueError('prepared recording arrays must have one non-empty common row count')
 if arrays['descriptor'].shape!=(n,DESCRIPTOR_DIM) or arrays['raw_waveforms'].shape!=(n,WINDOW) or arrays['legacy_normalized_waveforms'].shape!=(n,WINDOW):
  raise ValueError('prepared recording must contain N x D9 and two N x 64 waveform forms')
 expected_legacy=legacy_peak_normalize(arrays['raw_waveforms'])
 if not np.array_equal(arrays['legacy_normalized_waveforms'],expected_legacy):
  raise ValueError('legacy_normalized_waveforms must be the exact row-wise normalization of raw_waveforms')
 fit_idx,cal_idx,test_idx=stable_three_way_split(arrays['times'])
 source=frozen_uniform_candidate_source(arrays['descriptor'][fit_idx],arrays['labels'][fit_idx],arrays['descriptor'][cal_idx],arrays['labels'][cal_idx],arrays['descriptor'][test_idx])
 candidates=[np.asarray(row).copy() for row in source['candidates']]
 test_labels,test_wave=arrays['labels'][test_idx],arrays['raw_waveforms'][test_idx]
 support=candidate_support(candidates,test_labels)
 counts=np.asarray([len(row) for row in candidates],dtype=np.int64)
 source['level1_c0_reuse_events']=int((counts==0).sum())
 baselines=CausalTemporalBaselines().fit(arrays['raw_waveforms'][fit_idx],arrays['labels'][fit_idx])
 if not np.array_equal(baselines.transform(test_wave,'legacy_peak_normalized_float64'),arrays['legacy_normalized_waveforms'][test_idx]):
  raise ValueError('baseline legacy report must consume the prepared frozen legacy normalization')
 baseline_row,baseline_predictions=_baseline_columns(baselines,test_wave,candidates,test_labels,source)
 source_digest=sha256_json(_json_value({'units':source['model'].units_,'centroids':source['model'].centroids_,'radii':source['radii'],'candidate_lists':candidates,'all_rows':source['all_rows'],'level1_argmin_predictions':source['level1_argmin_predictions'],'level1_counters':{key:value for key,value in source.items() if key.startswith('level1_source_')},'calibration_counts':source['calibration_counts'],'fallback_mask':source['calibration_fallback_mask'],'radius_source':source['radius_source'],'payload54':source['payload54']}))
 provenance=dict(prepared.get('provenance',{}))
 split={'n_fit_events':int(fit_idx.size),'n_calibration_events':int(cal_idx.size),'n_test_events':int(test_idx.size),'fit_last_timestamp':int(arrays['times'][fit_idx][-1]),'calibration_first_timestamp':int(arrays['times'][cal_idx][0]),'calibration_last_timestamp':int(arrays['times'][cal_idx][-1]),'test_first_timestamp':int(arrays['times'][test_idx][0])}
 common={'dataset':record.dataset_id,'family':record.family,'source_input_path':record.source_input_path,'source_file_sha256':record.source_file_sha256,'source_input_subset_digest':record.source_input_subset_digest,'temporal_input_contract':TEMPORAL_INPUT_CONTRACT,'temporal_input_contract_sha256':TEMPORAL_INPUT_CONTRACT_SHA256,'repository_base_revision':snapshot['repository_base_revision'],'implementation_source_sha256':snapshot['implementation_source_sha256'],'implementation_source_digest':snapshot['implementation_source_digest'],**split,'candidate_recall_hits':int(sum(y in row for y,row in zip(test_labels,candidates))),'candidate_recall':support['candidate_recall'],'c0_events':support['c0_events'],'c1_events':support['c1_events'],'ambiguous_events':support['ambiguous_events'],'candidate_template_reads':int(counts[counts>1].sum()),'distinct_ambiguous_candidate_pairs':support['distinct_unordered_pairs'],'level1_source_scan_events':source['level1_source_scan_events'],'level1_source_unit_count':source['level1_source_unit_count'],'level1_source_descriptor_quantizations':source['level1_source_descriptor_quantizations'],'level1_source_centroid_reads':source['level1_source_centroid_reads'],'level1_source_l1_absolute_differences':source['level1_source_l1_absolute_differences'],'level1_source_l1_reduction_additions':source['level1_source_l1_reduction_additions'],'level1_source_radius_threshold_comparisons':source['level1_source_radius_threshold_comparisons'],'level1_source_argmin_reduction_comparisons':source['level1_source_argmin_reduction_comparisons'],'level1_c0_reuse_events':int((counts==0).sum()),'candidate_source_digest':source_digest,'candidate_source_summary':{'bits':BITS,'mode':'l1','radius_percentile':99.9,'all_rows_only':True,'calibration_counts':source['calibration_counts'],'calibration_fallback_mask':source['calibration_fallback_mask'],'radius_source':source['radius_source'],'radii':source['radii'],'payload54':source['payload54']},'fit_event_digest':_event_digest(prepared,fit_idx),'calibration_event_digest':_event_digest(prepared,cal_idx),'test_event_digest':_event_digest(prepared,test_idx),'raw_input_provenance_digest':sha256_json(_json_value(provenance)),'teacher_contract_hash':str(baselines.payload_accounting('fit_scale_float64')['teacher_input_contract_sha256']),'preparation_provenance':provenance,'baseline_prediction_digests':_prediction_digest(baseline_predictions),**baseline_row}
 rows=[]
 for delay in DELAYS:
  for feature_count in FEATURE_COUNTS:
   sketch=CausalTemporalSketch(delay_samples=delay,n_features=feature_count).fit(arrays['raw_waveforms'][fit_idx],arrays['labels'][fit_idx])
   primary_initial,diagnostics=sketch.assign_candidates(test_wave,candidates)
   primary=_with_spatial_fallback(primary_initial,candidates,source)
   ceiling=_with_spatial_fallback(sketch.same_horizon_float_assign(test_wave,candidates),candidates,source)
   metric,ceiling_metric=_prediction_metrics(primary,test_labels,candidates),_prediction_metrics(ceiling,test_labels,candidates)
   traffic=sketch.template_traffic(candidates)
   reads=int(np.asarray(diagnostics['logical_candidate_template_reads']).sum())
   sad_abs=int(np.asarray(diagnostics['logical_sad_absolute_differences']).sum())
   sad_adds=int(np.asarray(diagnostics['logical_sad_reduction_additions']).sum())
   diagnostic_bits=int(np.asarray(diagnostics['logical_candidate_template_bits']).sum())
   extraction={key:int(np.asarray(value).sum()) for key,value in sketch.logical_extraction_accounting(len(test_idx)).items()}
   category_events={'c0':int((counts==0).sum()),'c1':int((counts==1).sum()),'cgt1':int((counts>1).sum())}
   if any(total % len(test_idx) for total in extraction.values()):
    raise AssertionError('unconditional logical extraction totals must divide exactly by event count')
   per_event={key:value//len(test_idx) for key,value in extraction.items()}
   category_totals={category:{key:events*per_event[key] for key in extraction} for category,events in category_events.items()}
   category_totals['c0'].update({'template_reads':0,'template_bits':0,'sad_absolute_differences':0,'sad_reduction_additions':0})
   category_totals['c1'].update({'template_reads':0,'template_bits':0,'sad_absolute_differences':0,'sad_reduction_additions':0})
   category_totals['cgt1'].update({'template_reads':reads,'template_bits':diagnostic_bits,'sad_absolute_differences':sad_abs,'sad_reduction_additions':sad_adds})
   codes=sketch.transform(test_wave)
   ids=np.asarray(sketch.selected_filter_ids_,dtype=np.int64)
   bit_ok=bool(np.all((codes>=-15)&(codes<=15)) and np.all((sketch.templates_>=-15)&(sketch.templates_<=15)) and len(np.unique(ids))==feature_count and set(ids)<=set(sketch.eligible_filter_ids_) and np.all(np.asarray(diagnostics['winning_sad_distance'])<=30*feature_count))
   actual_bits=reads*5*feature_count
   level1_ok=bool(source['level1_source_scan_events']==len(test_idx) and source['level1_source_unit_count']==len(source['model'].units_) and source['level1_source_descriptor_quantizations']==len(test_idx)*DESCRIPTOR_DIM and source['level1_source_centroid_reads']==len(test_idx)*source['level1_source_unit_count'] and source['level1_source_l1_absolute_differences']==source['level1_source_centroid_reads']*DESCRIPTOR_DIM and source['level1_source_l1_reduction_additions']==source['level1_source_centroid_reads']*(DESCRIPTOR_DIM-1) and source['level1_source_radius_threshold_comparisons']==source['level1_source_centroid_reads'] and source['level1_source_argmin_reduction_comparisons']==len(test_idx)*max(source['level1_source_unit_count']-1,0) and source['level1_c0_reuse_events']==int((counts==0).sum()) and source['level1_argmin_predictions'].shape==(len(test_idx),))
   accounting_ok=bool(level1_ok and reads==int(counts[counts>1].sum()) and sad_abs==reads*feature_count and sad_adds==reads*(feature_count-1) and diagnostic_bits==reads*5*feature_count and actual_bits==int(traffic['actual_sketch_template_bits']) and extraction['logical_raw_prefix_samples_consumed']==len(test_idx)*sketch.prefix_length and extraction['logical_sample_quantizations']==len(test_idx)*sketch.prefix_length and extraction['logical_positive_coefficient_updates']+extraction['logical_negative_coefficient_updates']==extraction['logical_selected_feature_accumulator_updates'] and extraction['logical_normalization_shifts']==len(test_idx)*feature_count and extraction['logical_emitted_feature_codes']==len(test_idx)*feature_count and int(traffic['global_full64x5_template_bits'])==len(test_idx)*len(sketch.units_)*64*5 and np.all(np.asarray(diagnostics['logical_candidate_template_reads'])[counts<=1]==0) and all(sum(category_totals[category][key] for category in category_events)==extraction[key] for key in extraction) and sum(category_totals[category]['template_reads'] for category in category_events)==reads and sum(category_totals[category]['template_bits'] for category in category_events)==diagnostic_bits and sum(category_totals[category]['sad_absolute_differences'] for category in category_events)==sad_abs and sum(category_totals[category]['sad_reduction_additions'] for category in category_events)==sad_adds)
   scale_audit={'scale':sketch.scale_,'scale_exponent':sketch.scale_exponent_,'fit_abs_q999':sketch.fit_abs_q999_,'test_sample_saturation_fraction':sketch.sample_saturation_fraction(test_wave),'sketch_saturation':sketch.saturation_accounting(),'baseline_scale':baselines.scale_,'baseline_scale_exponent':baselines.scale_exponent_,'baseline_fit_abs_q999':baselines.fit_abs_q999_,'baseline_test_sample_saturation_fraction':baselines.test_sample_saturation_fraction(test_wave),'baseline_saturation':baselines.saturation_accounting()}
   row={**common,'delay_samples':delay,'feature_count':feature_count,'configuration':f'delay_{delay:02d}_m_{feature_count:02d}','overall_correct':metric['overall_correct'],'overall_accuracy':metric['overall_accuracy'],'ambiguous_correct':metric['ambiguous_correct'],'ambiguous_accuracy':metric['ambiguous_accuracy'],'per_unit_metrics':metric['per_unit'],'worst_unit_accuracy':metric['worst_unit_accuracy'],'same_horizon_float_ceiling_overall_correct':ceiling_metric['overall_correct'],'same_horizon_float_ceiling_ambiguous_correct':ceiling_metric['ambiguous_correct'],'same_horizon_float_ceiling_overall_accuracy':ceiling_metric['overall_accuracy'],'same_horizon_float_ceiling_ambiguous_accuracy':ceiling_metric['ambiguous_accuracy'],'full_teacher_overall_correct':baseline_row['full_teacher_overall_correct'],'full_teacher_ambiguous_correct':baseline_row['full_teacher_ambiguous_correct'],'global_64x5_template_bits':int(traffic['global_full64x5_template_bits']),'actual_template_bits':actual_bits,'logical_candidate_template_reads':reads,'logical_candidate_template_bits':actual_bits,'logical_sad_absolute_differences':sad_abs,'logical_sad_reduction_additions':sad_adds,'logical_extraction_counters':extraction,'logical_category_decomposition':category_totals,'memory_accounting':sketch.memory_accounting(),'scale_audit':scale_audit,'fisher_audit':_fisher_audit(sketch),'selected_filter_ids':ids.tolist(),'scale_exponent':int(sketch.scale_exponent_),'raw_input_provenance_ok':provenance.get('temporal_input')=='filtered_pre_event_normalization_main_channel','row_identity_ok':bool(_event_digest(prepared,np.arange(n))==provenance.get('row_digest')),'future_sample_isolation_ok':assert_future_tail_invariance(sketch,test_wave,candidates),'bit_width_invariants_ok':bit_ok,'accounting_invariants_ok':accounting_ok}
   rows.append(_json_value(row))
 return rows


def run_dataset(dataset: Dataset, *, record: PilotRecordSpec, implementation_snapshot: Mapping[str,Any] | None=None) -> list[dict[str,Any]]:
 """Prepare one recording once, then run its fixed causal temporal grid."""
 return run_prepared_recording(prepare_causal_events(dataset),record=record,implementation_snapshot=implementation_snapshot)


def _csv_cell(value: Any) -> Any:
 value=_json_value(value)
 return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False) if isinstance(value,(dict,list)) else value


def write_rows_csv(path: Path, rows: Sequence[Mapping[str,Any]]) -> None:
 """Write a deterministic inspection CSV while retaining rich rows in memory."""
 path.parent.mkdir(parents=True,exist_ok=True); names=sorted({str(key) for row in rows for key in row})
 with path.open('w',newline='',encoding='utf-8') as handle:
  writer=csv.DictWriter(handle,fieldnames=names); writer.writeheader()
  for row in rows: writer.writerow({name:_csv_cell(row.get(name)) for name in names})
 expected=[{name:('' if _csv_cell(row.get(name)) is None else str(_csv_cell(row.get(name)))) for name in names} for row in rows]
 with path.open(newline='',encoding='utf-8') as handle:
  round_trip=list(csv.DictReader(handle))
 if len(round_trip)!=len(rows) or (round_trip and list(round_trip[0])!=names) or round_trip!=expected:
  raise AssertionError('deterministic CSV round-trip/schema validation failed')


LOCKED_PILOT_FILE_SHA256=LOCKED_PILOT_SOURCE_SHA256


def _load_verified_path(path: Path, expected_sha256: str, loader: Callable[[Path], Dataset]) -> Dataset:
 """Re-hash the exact discovered file immediately before its lazy load."""
 if _sha256_file(path).lower()!=expected_sha256.lower():
  raise ValueError(f'pilot source changed before load: {path}')
 dataset=loader(path)
 if _sha256_file(path).lower()!=expected_sha256.lower():
  del dataset
  raise ValueError(f'pilot source changed during load: {path}')
 return dataset


def discover_pilot_datasets(*, duration_s: float=60.0, mearec_npz_dir: Path=DEFAULT_NPZ_DIR) -> dict[str,list[PilotRecordSpec]]:
 """Return verified lazy loaders for exactly the four pilot recordings."""
 found:dict[str,list[PilotRecordSpec]]={'hj':[],'mearec':[]}
 wanted={family:set(values) for family,values in PILOT_DATASET_IDS.items()}
 for scene in list_hybrid_janelia_scenes(ready_only=True):
  key=scene.get('short_name') or scene.get('scene_key')
  if key:
   dataset_id=f'hybrid_janelia_{str(key).replace("_filtered_gt","")}'
   path=Path(scene.get('npz_path',''))
   resolved=path.resolve(strict=True) if path.exists() else None
   if dataset_id in wanted['hj'] and resolved is not None and _sha256_file(resolved)==LOCKED_PILOT_FILE_SHA256[dataset_id]:
    expected=LOCKED_PILOT_FILE_SHA256[dataset_id]
    found['hj'].append(PilotRecordSpec(dataset_id=dataset_id,family='hj',source_input_path=str(resolved),source_file_sha256=expected,source_input_subset_digest=LOCKED_INPUT_SUBSET_SHA256,load_dataset=lambda resolved=resolved,expected=expected:_load_verified_path(resolved,expected,lambda verified:load_hybrid_janelia(path=verified,duration_s=duration_s))))
 for path in discover_npz(Path(mearec_npz_dir)):
  resolved=Path(path).resolve(strict=True)
  dataset_id=f'mearec_{path.stem}'
  if dataset_id in wanted['mearec'] and _sha256_file(resolved)==LOCKED_PILOT_FILE_SHA256[dataset_id]:
   expected=LOCKED_PILOT_FILE_SHA256[dataset_id]
   found['mearec'].append(PilotRecordSpec(dataset_id=dataset_id,family='mearec',source_input_path=str(resolved),source_file_sha256=expected,source_input_subset_digest=LOCKED_INPUT_SUBSET_SHA256,load_dataset=lambda resolved=resolved,expected=expected:_load_verified_path(resolved,expected,lambda verified:load_mearec_npz(verified,duration_s=duration_s))))
 for family in ('hj','mearec'):
  ids=[item.dataset_id for item in found[family]]
  if len(ids)!=2 or set(ids)!=wanted[family]: raise ValueError(f'--pilot requires exactly the two locked {family} pilot IDs; found {ids}')
 return found


def main(argv: Sequence[str] | None=None) -> None:
 parser=argparse.ArgumentParser(description='Fixed four-recording causal temporal-sketch pilot')
 parser.add_argument('--pilot',action='store_true',help='run only the locked four-recording development pilot')
 parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'output'/'causal_temporal_sketch_pilot')
 parser.add_argument('--duration',type=float,default=60.0)
 parser.add_argument('--mearec-npz-dir',type=Path,default=DEFAULT_NPZ_DIR,help=argparse.SUPPRESS)
 args=parser.parse_args(argv)
 if not args.pilot: parser.error('only --pilot is implemented; confirmation is deliberately unavailable')
 if args.duration!=60.0: parser.error('--pilot duration is locked to exactly 60 seconds')
 snapshot=implementation_source_snapshot()
 discovered=discover_pilot_datasets(duration_s=args.duration,mearec_npz_dir=args.mearec_npz_dir); rows_by_family={'hj':[],'mearec':[]}
 for family in ('hj','mearec'):
  for record in discovered[family]:
   dataset=record.load_dataset()
   try: rows_by_family[family].extend(run_dataset(dataset,record=record,implementation_snapshot=snapshot))
   finally: del dataset
  if len(rows_by_family[family])!=30: raise AssertionError('each pilot family must emit exactly 2 x 15 rows')
 validate_development_rows(rows_by_family)
 csv_paths={family:args.output_dir/f'causal_temporal_sketch_{family}_pilot.csv' for family in ('hj','mearec')}
 for family in ('hj','mearec'): write_rows_csv(csv_paths[family],rows_by_family[family])
 manifest=build_selection_manifest(rows_by_family=rows_by_family,csv_paths=csv_paths)
 manifest_path=args.output_dir/'causal_temporal_sketch_selection.json'
 write_json_deterministic(manifest_path,manifest)
 print(f'pilot gate_pass={manifest["gate"]["gate_pass"]} manifest_sha256={_sha256_file(manifest_path)}')
 print('confirmation was not run and is not implemented by this entry point')


if __name__=='__main__': main()
