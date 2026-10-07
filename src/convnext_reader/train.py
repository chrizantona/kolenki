"""Kaggle fp16 port of goodpj's reader with resumable warm-start fine-tuning.

Default requested experiment: all4349 weak studies, gold58 excluded from gradients,
publicfold0 warmstart, two fixed epochs, encoder1e-5/head5e-5, effective study batch8.
This is additional training, not a reproduction of public0.944 or clean weak OOF.
Use --pilot-steps8 for the separate throughput pilot; never resume a pilot into full training.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from knee import LABELS, KneeNet, StudyDataset, build_study_table
from zip_cache_loader import MixedVolumeLoader as VolumeLoader, load_cache_mixed as load_cache

CONFIG_SCHEMA = 1
FAILURE_CONTEXT = {}


def serializable(value):
    if isinstance(value, Path):return str(value)
    if isinstance(value, dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [serializable(v) for v in value]
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(map(str,ids))).encode()).hexdigest()


def atomic_torch(payload,path):
    torch.save(payload,str(path)+'.tmp')
    os.replace(str(path)+'.tmp',path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


class SeededStudyDataset(StudyDataset):
    """Equivalent random-slot/window distributions, deterministic by study+epoch.

    Allows a resumed run to change worker scheduling without changing CPU window
    choices. GPU augmentation/dropout RNG is stored at completed optimizer groups.
    """
    def __init__(self,*args,epoch=0,seed=0,**kwargs):
        super().__init__(*args,**kwargs);self.epoch=epoch;self.seed=seed
    def __getitem__(self,i):
        state=np.random.get_state()
        digest=hashlib.blake2b(f'{self.seed}:{self.epoch}:{self.studies[i]}'.encode(),digest_size=4).digest()
        np.random.seed(int.from_bytes(digest,'little'))
        try:return super().__getitem__(i)
        finally:np.random.set_state(state)


class ResumableBatchSampler:
    def __init__(self,n,bs,accum,seed,epoch,start_group=0):
        order=np.random.RandomState(seed+epoch).permutation(n)
        effective=bs*accum
        self.indices=order[:len(order)//effective*effective]
        self.bs=bs;self.accum=accum;self.start_group=start_group
    def __iter__(self):
        start=self.start_group*self.bs*self.accum
        for i in range(start,len(self.indices),self.bs):yield self.indices[i:i+self.bs].tolist()
    def __len__(self):
        return len(self.indices)//self.bs-self.start_group*self.accum


@torch.no_grad()
def predict(model,loader,res,device,amp_dtype):
    model.eval()
    rows=[]
    for x,mask,pos,_ in loader:
        x=x.to(device,non_blocking=True);mask=mask.to(device);pos=pos.to(device)
        with torch.autocast('cuda',dtype=amp_dtype):
            rows.append(model(x,mask,pos,res=res).float().cpu())
    return torch.cat(rows).numpy()


def score_gold(y,p):
    per=[float(roc_auc_score(y[:,i],p[:,i])) if np.unique(y[:,i]).size==2 else None
         for i in range(y.shape[1])]
    return {'macro_auc':float(np.mean([v for v in per if v is not None])),
            'per_class_auc':dict(zip(LABELS,per)),'n':len(y)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--labels',type=Path,required=True)
    ap.add_argument('--cache-roots',type=Path,nargs='+',required=True)
    ap.add_argument('--cache-fingerprints',type=Path,required=True)
    ap.add_argument('--init-ckpt',type=Path,required=True)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('/kaggle/working/run'))
    ap.add_argument('--backbone',default='convnext_tiny')
    ap.add_argument('--fold',type=int,default=-1)
    ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--bs',type=int,default=1)
    ap.add_argument('--accum',type=int,default=8)
    ap.add_argument('--res',type=int,default=256)
    ap.add_argument('--k',type=int,default=8)
    ap.add_argument('--k-eval',type=int,default=12)
    ap.add_argument('--lr',type=float,default=1e-5)
    ap.add_argument('--lr-head',type=float,default=5e-5)
    ap.add_argument('--wd',type=float,default=.02)
    ap.add_argument('--ema',type=float,default=.998)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--pilot-steps',type=int,default=0,help='Optimizer groups; partial-cache pilot only.')
    ap.add_argument('--max-seconds',type=float,default=3*3600)
    ap.add_argument('--checkpoint-seconds',type=float,default=900)
    ap.add_argument('--freeze-encoder',action='store_true')
    ap.add_argument('--gradient-checkpointing',action='store_true',help='Only enable after memory pilot requires it.')
    ap.add_argument('--amp',choices=['fp16','bf16'],default='fp16')
    args=ap.parse_args()
    assert args.bs>0 and args.accum>0 and args.epochs>0
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no CPU full-training fallback.')
    if args.amp=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('GPU does not support bf16.')
    start=time.monotonic()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    FAILURE_CONTEXT.update(output_dir=str(args.output_dir),start=start,pilot_only=bool(args.pilot_steps),
        planned_epochs=args.epochs)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device('cuda')
    amp_dtype=torch.float16 if args.amp=='fp16' else torch.bfloat16
    lab=pd.read_csv(args.labels,index_col=0)
    official=pd.read_csv(args.data_dir/'train.csv').set_index('StudyInstanceUID')
    if not lab.index.is_unique or set(lab.index)!=set(official.index):
        raise RuntimeError('Consensus table study identities differ from official training data.')
    if not lab.is_gold.isin([0,1]).all():raise RuntimeError('Invalid gold-annotation flag.')
    gold_ids=set(official.index[official[LABELS].notna().all(axis=1)])
    if set(lab.index[lab.is_gold.eq(1)])!=gold_ids:raise RuntimeError('Gold exclusion identity mismatch.')
    paths,counts,cache_summary,cache_fingerprints=load_cache(args.cache_roots)
    expected_cache_fingerprints=json.loads(args.cache_fingerprints.read_text())
    if len(cache_fingerprints)!=7 or cache_fingerprints!=expected_cache_fingerprints:
        raise RuntimeError('Mounted cache receipts differ from independently verified input fingerprints.')
    expected_preprocess=json.loads(Path(__file__).with_name('source_manifest.json').read_text())['preprocess.py']['sha256']
    if cache_summary[0]['preprocess_sha256']!=expected_preprocess:
        raise RuntimeError('Cache preprocessing differs from the exact published source.')
    se=pd.read_csv(args.data_dir/'train_series.csv')
    table=build_study_table(se,counts)
    weak=lab[lab.is_gold.eq(0)]
    tr=weak[weak.fold.ne(args.fold)] if args.fold>=0 else weak
    expected_by_study=se.groupby('StudyInstanceUID').SeriesInstanceUID.agg(set).to_dict()
    complete_ids={st for st,srs in expected_by_study.items() if srs.issubset(paths)}
    if args.pilot_steps:
        tr=tr[tr.index.isin(complete_ids)]
        if len(tr)<args.bs*args.accum:raise RuntimeError('Not enough fully cached weak studies for pilot.')
    elif not set(tr.index).issubset(complete_ids) or not all(s['complete'] for s in cache_summary):
        raise RuntimeError('Full training requires all selected studies and completed cache shards.')
    assert not set(tr.index)&gold_ids,'Gold studies found in gradient training.'
    if not tr[LABELS].notna().all().all():raise RuntimeError('Source consensus contains unknown targets.')
    if not tr[LABELS].apply(lambda col:col.between(0,1)).all().all():
        raise RuntimeError('Weak targets are not finite probabilities in [0,1].')
    if any(not any(table.get(st,[])) for st in tr.index):raise RuntimeError('A selected study has no usable series.')
    provenance={'experiment':f'fullweak{args.epochs}epwarmstart' if args.fold==-1 else f'warmstartfoldtraining{args.epochs}ep',
        'planned_epochs':args.epochs,
        'pilot_only':bool(args.pilot_steps),'train_studies':len(tr),'train_ids_sha256':ids_hash(tr.index),
        'gold_excluded':len(gold_ids),'gold_ids_sha256':ids_hash(gold_ids),
        'labels_sha256':sha(args.labels),'init_checkpoint_sha256':sha(args.init_ckpt),
        'model_source_sha256':sha(Path(__file__).with_name('knee.py')),
        'training_script_sha256':sha(__file__),'config_schema':CONFIG_SCHEMA,
        'cache_loader_source_sha256':sha(Path(__file__).with_name('zip_cache_loader.py')),
        'official_train_sha256':sha(args.data_dir/'train.csv'),
        'official_series_sha256':sha(args.data_dir/'train_series.csv'),
        'cache_shard_fingerprints':cache_fingerprints,
        'cache_preprocess_sha256':cache_summary[0]['preprocess_sha256'],
        'weak_oof_clean':False,'author_fold_uid_manifest_verified':False,
        'notes':'Fixed-final EMA, no checkpoint selection by gold. Warmstart folds are not certified OOF.'}
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2))
    pd.Series(tr.index,name='StudyInstanceUID').to_csv(args.output_dir/'gradient_train_ids.csv',index=False)
    pd.Series(sorted(gold_ids),name='StudyInstanceUID').to_csv(args.output_dir/'excluded_gold_ids.csv',index=False)
    print(json.dumps({'args':serializable(vars(args)),
                      'provenance':provenance,'gpu':torch.cuda.get_device_name(0)}),flush=True)
    init=torch.load(args.init_ckpt,map_location='cpu',weights_only=False)
    model=KneeNet(args.backbone,pretrained=False)
    model.load_state_dict(init['model'],strict=True)
    model=model.to(device).to(memory_format=torch.channels_last)
    if args.gradient_checkpointing:
        if not hasattr(model.enc,'set_grad_checkpointing'):
            raise RuntimeError('This timm encoder does not support gradient checkpointing.')
        model.enc.set_grad_checkpointing(True)
    if args.freeze_encoder:
        for p in model.enc.parameters():p.requires_grad_(False)
    ema=copy.deepcopy(model).eval()
    for p in ema.parameters():p.requires_grad_(False)
    enc=[p for p in model.enc.parameters() if p.requires_grad]
    head=[p for n,p in model.named_parameters() if not n.startswith('enc.') and p.requires_grad]
    param_groups=([{'params':enc,'lr':args.lr}] if enc else [])+[{'params':head,'lr':args.lr_head}]
    opt=torch.optim.AdamW(param_groups,weight_decay=args.wd)
    groups_per_epoch=len(tr)//(args.bs*args.accum)
    planned_steps=args.epochs*groups_per_epoch
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:
        min(1.,(s+1)/groups_per_epoch)*.5*(1+math.cos(math.pi*min(1.,s/planned_steps))))
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp=='fp16')
    loader=VolumeLoader(paths)
    epoch,cursor,step=0,0,0
    state_path=args.output_dir/f'state_fold{args.fold}.pt'
    check_path=args.output_dir/f'fold{args.fold}.pt'
    resume=args.resume or (state_path if state_path.exists() else None)
    history=[]
    if resume:
        st=torch.load(resume,map_location='cpu',weights_only=False)
        for key in ['train_ids_sha256','labels_sha256','init_checkpoint_sha256','model_source_sha256',
                    'training_script_sha256','config_schema','official_train_sha256','official_series_sha256',
                    'cache_preprocess_sha256','cache_shard_fingerprints','cache_loader_source_sha256']:
            if st['provenance'][key]!=provenance[key]:raise RuntimeError(f'Resume provenance differs: {key}')
        core_keys=['backbone','fold','epochs','bs','accum','res','k','k_eval','lr','lr_head','wd','ema',
                   'freeze_encoder','gradient_checkpointing','seed','amp','pilot_steps','checkpoint_seconds']
        for key in core_keys:
            if st['args'][key]!=vars(args)[key]:raise RuntimeError(f'Resume training configuration differs: {key}')
        model.load_state_dict(st['model']);ema.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt']);sched.load_state_dict(st['sched']);scaler.load_state_dict(st['scaler'])
        epoch,cursor,step=st['next_epoch'],st['next_group'],st['step']
        history=st.get('history',[])
        restore_rng(st['rng'])
        print(f'Resumed epoch={epoch},group={cursor},optimizer_step={step}',flush=True)
    last_checkpoint=time.monotonic();reason='fixed_epochs_complete';group_seconds=[]
    gold=lab[lab.is_gold.eq(1)]
    gold_available=gold[gold.index.isin(complete_ids)]
    def save(next_epoch,next_group):
        serial_args=serializable(vars(args))
        atomic_torch({'model':model.state_dict(),'ema':ema.state_dict(),'opt':opt.state_dict(),
            'sched':sched.state_dict(),'scaler':scaler.state_dict(),'step':step,
            'next_epoch':next_epoch,'next_group':next_group,'rng':rng_state(),
            'args':serial_args,'provenance':provenance,'history':history},state_path)
        FAILURE_CONTEXT.update(last_safe_state=str(state_path),
            safe_next_epoch=next_epoch,safe_next_group=next_group,safe_optimizer_steps=step,
            safe_attempted_optimizer_groups=next_epoch*groups_per_epoch+next_group)
        atomic_torch({'model':ema.state_dict(),'args':serial_args,'provenance':provenance,
            'training_progress':{'optimizer_steps':step,'next_epoch':next_epoch,
                'next_group':next_group,'pilot_only':bool(args.pilot_steps)}},check_path)
        FAILURE_CONTEXT.update(last_safe_export=str(check_path),export_safe_optimizer_steps=step,
            export_safe_next_epoch=next_epoch,export_safe_next_group=next_group)
    # Retain an initial safe state even if the first microbatch runs out of GPU memory.
    save(epoch,cursor)
    max_valid_slots=0;max_valid_views_per_microbatch=0
    while epoch<args.epochs:
        if args.pilot_steps and epoch*groups_per_epoch+cursor>=args.pilot_steps:
            reason='pilot_steps_complete';save(epoch,cursor);break
        dataset=SeededStudyDataset(tr.index,table,'',tr[LABELS].values.astype(np.float32),
            args.k,train=True,loader=loader,epoch=epoch,seed=args.seed)
        sampler=ResumableBatchSampler(len(tr),args.bs,args.accum,args.seed,epoch,cursor)
        dl=torch.utils.data.DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers,
            pin_memory=True,persistent_workers=False)
        iterator=iter(dl);model.train()
        if args.freeze_encoder:model.enc.eval()
        total_loss=0.;epoch_t0=time.monotonic();processed=0;views=0
        while cursor<groups_per_epoch:
            reserve=max(60.,2*np.median(group_seconds[-8:])) if group_seconds else 60.
            if time.monotonic()-start>=args.max_seconds-reserve:
                reason='walltime_budget';break
            t0=time.monotonic();opt.zero_grad(set_to_none=True);group_loss=0.
            for _ in range(args.accum):
                x,mask,pos,y=next(iterator)
                views+=int(mask.sum())*args.k
                max_valid_slots=max(max_valid_slots,int(mask.sum(dim=1).max()))
                max_valid_views_per_microbatch=max(max_valid_views_per_microbatch,int(mask.sum())*args.k)
                FAILURE_CONTEXT.update(max_valid_slots_per_study=max_valid_slots,
                    max_valid_views_per_microbatch=max_valid_views_per_microbatch)
                x=x.to(device,non_blocking=True);mask=mask.to(device);pos=pos.to(device);y=y.to(device)
                with torch.autocast('cuda',dtype=amp_dtype):
                    logits=model(x,mask,pos,res=args.res,train=True)
                    loss=F.binary_cross_entropy_with_logits(logits.float(),y)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at epoch{epoch} group{cursor}')
                scaler.scale(loss/args.accum).backward();group_loss+=float(loss.detach())/args.accum
            scaler.unscale_(opt)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
            if args.amp=='bf16' and not torch.isfinite(grad_norm):
                raise RuntimeError('Nonfinite bf16 gradients: optimizer update refused.')
            old_scale=scaler.get_scale();scaler.step(opt);scaler.update()
            updated=scaler.get_scale()>=old_scale
            if updated:
                sched.step();step+=1
                decay=min(args.ema,(1+step)/(10+step))
                with torch.no_grad():
                    for pe,pm in zip(ema.parameters(),model.parameters()):pe.lerp_(pm,1-decay)
                    for be,bm in zip(ema.buffers(),model.buffers()):be.copy_(bm)
            cursor+=1;processed+=1;total_loss+=group_loss
            FAILURE_CONTEXT.update(observed_completed_optimizer_groups=epoch*groups_per_epoch+cursor,
                observed_successful_optimizer_updates=step)
            torch.cuda.synchronize();group_seconds.append(time.monotonic()-t0)
            if cursor%8==0 or cursor==groups_per_epoch:
                print(json.dumps({'epoch':epoch,'group':cursor,'of':groups_per_epoch,
                    'loss':total_loss/max(processed,1),'sec_per_group_recent':float(np.median(group_seconds[-8:])),
                    'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
                    'optimizer_step':step}),flush=True)
            if time.monotonic()-last_checkpoint>=args.checkpoint_seconds:
                save(epoch,cursor);last_checkpoint=time.monotonic()
            if args.pilot_steps and epoch*groups_per_epoch+cursor>=args.pilot_steps:
                reason='pilot_steps_complete';break
        row={'epoch':epoch,'processed_optimizer_groups':processed,'mean_loss':total_loss/max(processed,1),
             'seconds':time.monotonic()-epoch_t0,'valid_views':views}
        if reason!='fixed_epochs_complete':
            history.append(row);save(epoch,cursor);break
        # Fixed epoch boundary; do not select checkpoint by gold diagnostics.
        history.append(row)
        epoch+=1;cursor=0;save(epoch,0);last_checkpoint=time.monotonic()
        if len(gold_available)==58 and time.monotonic()-start<args.max_seconds-300:
            gdl=torch.utils.data.DataLoader(StudyDataset(gold_available.index,table,'',None,
                args.k_eval,loader=loader),batch_size=args.bs,num_workers=args.workers,pin_memory=True)
            predictions=predict(ema,gdl,args.res,device,amp_dtype)
            row['gold_diagnostic']=score_gold(gold_available[[f'gold_{c}' for c in LABELS]].values,predictions)
            pd.DataFrame(predictions,index=gold_available.index,columns=LABELS).to_csv(
                args.output_dir/f'gold_logits_epoch{epoch}.csv')
            print(json.dumps(row['gold_diagnostic']),flush=True)
    if not state_path.exists():save(epoch,cursor)
    report={'reason':reason,'planned_epochs':args.epochs,'epochs_completed':epoch,'next_group':cursor,'optimizer_steps':step,
        'elapsed_seconds':time.monotonic()-start,'history':history,'pilot_only':bool(args.pilot_steps),
        'peak_allocated_GB':torch.cuda.max_memory_allocated()/1e9,
        'max_valid_slots_per_study':max_valid_slots,
        'max_valid_views_per_microbatch':max_valid_views_per_microbatch,
        'median_optimizer_group_seconds':float(np.median(group_seconds)) if group_seconds else None,
        'full_epoch_groups':groups_per_epoch,
        'selected_training_studies':len(tr),
        'studies_per_epoch_processed':groups_per_epoch*args.bs*args.accum,
        'studies_per_epoch_dropped':len(tr)-groups_per_epoch*args.bs*args.accum,
        'projected_compute_epoch_seconds':float(np.median(group_seconds)*groups_per_epoch) if group_seconds else None,
        'projection_excludes_validation_checkpoint_IO_and_variability':True,
        'gold_diagnostics_N':len(gold_available),'training_complete':epoch>=args.epochs and not args.pilot_steps and step>0,
        'useful_optimizer_updates':step>0,'environment':{'python':platform.python_version(),
            'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__}}
    report['attempted_optimizer_groups']=epoch*groups_per_epoch+cursor
    report['amp_skipped_optimizer_updates']=report['attempted_optimizer_groups']-step
    (args.output_dir/'training_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    if step==0:raise SystemExit('No successful optimizer updates; exported weights are not a trained candidate.')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if not FAILURE_CONTEXT:raise
        # An interrupted accumulation group is not a safe resume boundary. Preserve
        # the last atomically saved state/export; never serialize partial gradients.
        context=dict(FAILURE_CONTEXT)
        out=Path(context.pop('output_dir','/kaggle/working/run'));out.mkdir(parents=True,exist_ok=True)
        started=context.pop('start',time.monotonic())
        failure={'reason':'exception','exception_type':type(exc).__name__,'message':str(exc),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started,
            'training_complete':False,**context}
        failure['optimizer_steps']=context.get('safe_optimizer_steps',0)
        failure['attempted_optimizer_groups']=context.get('safe_attempted_optimizer_groups',0)
        failure['useful_optimizer_updates']=failure['optimizer_steps']>0
        failure['last_safe_checkpoint_present']=bool(context.get('last_safe_state')) and Path(context['last_safe_state']).is_file()
        failure['last_safe_export_present']=bool(context.get('last_safe_export')) and Path(context['last_safe_export']).is_file()
        if torch.cuda.is_available():failure['peak_allocated_GB']=torch.cuda.max_memory_allocated()/1e9
        (out/'failure.json').write_text(json.dumps(failure,indent=2))
        (out/'training_report.json').write_text(json.dumps(failure,indent=2))
        print(json.dumps({k:v for k,v in failure.items() if k!='traceback'}),flush=True)
        raise
