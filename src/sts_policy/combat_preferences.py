"""Provisional private-search preference targets over public-equivalent groups."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .combat_residual import equivalent_positions

TARGET_VERSION = "native_root_preferences_v1"


def preferences(encoded, panel, *, minimum_visits=8, minimum_conditions=4,
                minimum_gap=.15, agreement=.8, maximum_pairs=8):
    """Normalize visited root means within a search, then retain stable ordering.

    Search means are biased diagnostics, not calibrated public action values.
    Thresholds are pilot hypotheses, not confidence intervals. Equivalent copy
    groups share values; source/target distinctions remain intact.
    """
    groups=[]; covered=set()
    for position in range(len(encoded.action_indices)):
        if position not in covered:
            mask=equivalent_positions(encoded,position)
            groups.append(mask); covered.update(np.flatnonzero(mask).tolist())
    lookup={int(index):next(g for g,mask in enumerate(groups) if mask[i])
            for i,index in enumerate(encoded.action_indices)}
    conditions=[]
    for result in panel['results']:
        if result['status']!='ok':
            continue
        sums=np.zeros(len(groups));counts=np.zeros(len(groups))
        for edge in result['root_edges']:
            if edge['index'] not in lookup:
                raise ValueError('root edge is absent from public menu')
            if edge['visits']>=minimum_visits:
                group=lookup[edge['index']]
                sums[group]+=float(edge['mean_evaluation'])*edge['visits'];counts[group]+=edge['visits']
        values=np.full(len(groups),np.nan)
        selected=counts>0
        values[selected]=sums[selected]/counts[selected]
        if selected.sum()<2:
            continue
        low,high=float(values[selected].min()),float(values[selected].max())
        if not np.isfinite([low,high]).all() or high-low<1e-6:
            continue
        values[selected]=(values[selected]-low)/(high-low)
        conditions.append(values)
    if len(conditions)<minimum_conditions:
        return []
    matrix=np.stack(conditions);pairs=[]
    for first in range(len(groups)):
        for second in range(first+1,len(groups)):
            differences=matrix[:,first]-matrix[:,second]
            differences=differences[np.isfinite(differences)]
            if len(differences)<minimum_conditions:
                continue
            gap=float(differences.mean())
            if abs(gap)<minimum_gap:
                continue
            sign=1 if gap>0 else -1
            support=float((differences*sign>0).mean())
            if support<agreement:
                continue
            high_group,low_group=(first,second) if sign>0 else (second,first)
            pairs.append((groups[high_group],groups[low_group],abs(gap)*support))
    return sorted(pairs,key=lambda p:p[2],reverse=True)[:maximum_pairs]


def preference_loss(logits, pairs, *, hinge_margin=None):
    """Copy multiplicity cannot boost a group's score or target weight."""
    maximum=max(len(row) for row in pairs)
    directions=np.zeros((len(pairs),maximum,logits.shape[1]),np.float32)
    weights=np.zeros((len(pairs),maximum),np.float32)
    for i,row in enumerate(pairs):
        if not row:
            raise ValueError('every preference row requires a supported pair')
        for j,(high,low,weight) in enumerate(row):
            if not high.any() or not low.any() or weight<=0 or not np.isfinite(weight):
                raise ValueError('nonempty equivalent groups and positive finite weights required')
            directions[i,j,:len(high)]=high/high.sum()-low/low.sum()
            weights[i,j]=weight
    directions=torch.as_tensor(directions,device=logits.device)
    weights=torch.as_tensor(weights,device=logits.device)
    gaps=torch.einsum('bpa,ba->bp',directions,logits)
    penalties=F.softplus(-gaps) if hinge_margin is None else F.relu(hinge_margin-gaps)
    return ((penalties*weights).sum(-1)/weights.sum(-1)).mean()
