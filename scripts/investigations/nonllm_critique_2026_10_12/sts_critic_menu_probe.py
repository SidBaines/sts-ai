from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from sts_policy.encoding import encode_observation
from sts_policy.model_agent import collate_public,forward_policy,policy_from_payload
from sts_policy.training import load_checkpoint,file_identity

def main():
    torch.set_num_threads(1)
    rows=json.loads(Path('/private/tmp/nonllm_encoding_critique_results.json').read_text())['synthetic_counterexamples']
    row=next(r for r in rows if r['case']=='untargeted_menu_enemy_block')
    first,second=deepcopy(row['first_public_observation']),deepcopy(row['second_public_observation'])
    for obs in (first,second):
        obs['state']['combat']['cards']['hand'].append({'name':'Strike','type':'Attack','cost_for_turn':1})
        obs['legal_actions'].append({'index':3,'description':'play Strike [Attack] -> SPIKE_SLIME_M [enemy 0] (cost 1; deal 6)'})
    a,b=encode_observation(first),encode_observation(second)
    source=Path('/Users/sidbaines/.codex/worktrees/nonllm-curriculum/SlayTheSpireAI/data/nonllm_native_revision/campaign/joint_act3_0030.pt')
    payload=load_checkpoint(source);model=policy_from_payload(payload).cpu()
    with torch.no_grad():
        logits,values,mask=forward_policy(model,collate_public([first,second],'cpu'))
    out={'scope':'synthetic public-state architecture counterexample, no simulator rollout, backward or update','checkpoint':str(source),'sha256':file_identity(source),'model_config':payload['model_config'],'state_features_equal':bool(np.array_equal(a.state_features,b.state_features)),'action_features_equal':bool(np.array_equal(a.action_features,b.action_features)),'action_changed_coordinates':np.argwhere(a.action_features!=b.action_features).tolist(),'values':values.tolist(),'critic_values_equal':bool(values[0]==values[1]),'logits':logits.tolist(),'first_public_observation':first,'second_public_observation':second}
    Path('/private/tmp/sts_critic_menu_probe.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if 'public_observation' not in k}))
if __name__=='__main__':main()
