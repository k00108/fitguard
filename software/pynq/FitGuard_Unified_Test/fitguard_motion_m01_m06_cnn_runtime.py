"""Pure NumPy runtime for the six-subject range-fusion CNN candidate."""

from pathlib import Path
import json
import numpy as np


LABELS = ("Stable", "Mild Abnormality", "Marked Abnormality")


def _smooth(x):
    return np.convolve(x, np.asarray([0.2, 0.3, 0.3, 0.2]), mode="same")


def coherent_turn_present(vision, aux, left=0.14375, right=0.85625, speed=0.08):
    """Label-blind physical/trajectory-end turn detector."""
    vision=np.asarray(vision,dtype=np.float32); valid=np.asarray(aux[:,0]>0.5)
    if valid.sum()<12: return False
    cx=vision[:,0]; vx=_smooth(vision[:,5]); valid_cx=cx[valid]
    center=float(np.median(valid_cx)); low,high=np.quantile(valid_cx,[.10,.90])
    span=float(high-low); maximum=float(np.max(np.abs(valid_cx-center)))
    moving=np.where(np.abs(vx)>=speed*.625,np.sign(vx),0); moving=moving[moving!=0]
    changes=int(np.sum(moving[1:]!=moving[:-1])) if len(moving)>1 else 0
    if changes>2: return False
    for i in range(3,len(cx)-3):
        before=float(np.mean(vx[i-3:i])); after=float(np.mean(vx[i+1:i+4]))
        reversal=before*after<0 and min(abs(before),abs(after))>=speed
        physical=cx[i]<=left or cx[i]>=right
        endpoint=(span>=.18 and maximum>=.10 and abs(float(cx[i])-center)>=max(.10,.75*maximum)
                  and (cx[i]<=low or cx[i]>=high))
        if valid[i] and reversal and (physical or endpoint): return True
    return False


def _conv_relu(x, w, b):
    kernel=w.shape[0]; left=(kernel-1)//2; right=kernel-1-left
    padded=np.pad(x,((left,right),(0,0)))
    windows=np.stack([padded[i:i+kernel] for i in range(len(x))])
    return np.maximum(np.tensordot(windows,w,axes=([1,2],[0,1]))+b,0.0)


def _pool2(x):
    usable=len(x)-len(x)%2
    return x[:usable].reshape(usable//2,2,x.shape[1]).max(axis=1)


def _dense_relu(x,w,b): return np.maximum(x@w+b,0.0)


def canonicalize(imu,vision,aux):
    imu=np.asarray(imu,dtype=np.float32).copy(); vision=np.asarray(vision,dtype=np.float32).copy()
    aux=np.asarray(aux,dtype=np.float32); imu-=np.median(imu,axis=0,keepdims=True)
    valid=aux[:,0]>0.5
    if valid.any():
        baseline=np.median(vision[valid,:5],axis=0)
        vision[valid,:2]-=baseline[:2]
        vision[valid,2:5]=vision[valid,2:5]/np.maximum(baseline[2:5],1e-4)-1.0
    vision[~valid]=0.0
    acc=np.linalg.norm(imu[:,:3],axis=1); gyro=np.linalg.norm(imu[:,3:],axis=1)
    acc_jerk=np.linalg.norm(np.diff(imu[:,:3],axis=0),axis=1)
    gyro_jerk=np.linalg.norm(np.diff(imu[:,3:],axis=0),axis=1)
    if valid.any():
        vv=vision[valid]; speed=np.linalg.norm(vv[:,5:7],axis=1)
        speed_rms=np.sqrt(np.mean(speed*speed)); excursion=np.hypot(np.ptp(vv[:,0]),np.ptp(vv[:,1]))
    else: speed_rms=excursion=0.0
    ranges=np.asarray([np.sqrt(np.mean(acc*acc)),np.sqrt(np.mean(gyro*gyro)),np.max(acc),np.max(gyro),
                       np.sqrt(np.mean(acc_jerk*acc_jerk)),np.sqrt(np.mean(gyro_jerk*gyro_jerk)),
                       speed_rms,excursion],dtype=np.float32)
    imu8=np.concatenate([imu,acc[:,None],gyro[:,None]],axis=1)
    return imu8,vision,ranges


class RangeFusionCNN:
    def __init__(self,model_dir):
        model_dir=Path(model_dir)
        policy_path=model_dir/'deployment_policy.json'
        self.policy=json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.exists() else {}
        with np.load(model_dir/'range_fusion_v2_weights.npz',allow_pickle=False) as data:
            labels=tuple(data['labels'].astype(str).tolist())
            if labels!=LABELS: raise ValueError(f'Unexpected labels: {labels}')
            self.w={k:data[k].astype(np.float32) for k in data.files if k not in ('model_kind','labels','range_names')}
        with np.load(model_dir/'range_fusion_v2_normalization.npz',allow_pickle=False) as data:
            self.imu_mean=data['imu_mean'].astype(np.float32); self.imu_std=data['imu_std'].astype(np.float32)
            self.vis_mean=data['vision_mean'].astype(np.float32); self.vis_std=data['vision_std'].astype(np.float32)
            self.range_mean=data['range_mean'].astype(np.float32); self.range_std=data['range_std'].astype(np.float32)
        self.no_excursion=not bool(self.policy.get('vision_center_excursion_used_for_severity',True))
        guard=self.policy.get('turn_guard',{})
        self.turn_guard=bool(guard.get('only_adjust_close_mild_marked_decisions',False))
        self.turn_guard_margin=0.10

    def _branch(self,x,prefix):
        x=_conv_relu(x,self.w[f'{prefix}_conv1_0'],self.w[f'{prefix}_conv1_1']); x=_pool2(x)
        x=_conv_relu(x,self.w[f'{prefix}_conv2_0'],self.w[f'{prefix}_conv2_1']); x=_pool2(x)
        x=_conv_relu(x,self.w[f'{prefix}_conv3_0'],self.w[f'{prefix}_conv3_1']); x=x.mean(axis=0)
        return _dense_relu(x,self.w[f'{prefix}_dense_0'],self.w[f'{prefix}_dense_1'])

    def predict(self,imu,vision,aux):
        turn=coherent_turn_present(vision,aux) if self.turn_guard else False
        imu,vision,ranges=canonicalize(imu,vision,aux)
        if self.no_excursion: ranges[7]=0.0
        imu=np.clip((imu-self.imu_mean)/self.imu_std,-8,8)
        vision=np.clip((vision-self.vis_mean)/self.vis_std,-8,8)
        ranges=np.clip((ranges-self.range_mean)/self.range_std,-8,8)
        fi=self._branch(imu,'imu'); fv=self._branch(vision,'vision')
        fr=_dense_relu(ranges,self.w['range_dense1_0'],self.w['range_dense1_1'])
        fr=_dense_relu(fr,self.w['range_dense2_0'],self.w['range_dense2_1'])
        x=np.concatenate([fi,fv,fr]); x=_dense_relu(x,self.w['fusion_dense1_0'],self.w['fusion_dense1_1'])
        x=_dense_relu(x,self.w['fusion_dense2_0'],self.w['fusion_dense2_1'])
        logits=x@self.w['fusion_output_0']+self.w['fusion_output_1']; logits-=logits.max()
        p=np.exp(logits); p=(p/p.sum()).astype(np.float32)
        if turn and int(np.argmax(p))==2 and float(p[2]-p[1])<self.turn_guard_margin:
            p[1],p[2]=p[2],p[1]
        return p

    def classify(self,imu,vision,aux):
        p=self.predict(imu,vision,aux); i=int(np.argmax(p))
        return {'label':LABELS[i],'confidence':float(p[i]),'probabilities':p.tolist()}
