"""Create a dated, captioned plot gallery from an immutable snapshot of finished fights."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import statistics
import textwrap

from sts_ai import fight_lab as lab
from sts_ai.fight_analysis import analyze, cluster_interval
from sts_ai.provenance import file_sha256


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'data/experiments/single_fight_v1/search_10min_2026-09-06/runs/gemma4_t1_p095_k64_v1'
COLORS = {'win': '#23876c', 'loss': '#cb4d51', 'incomplete': '#da9a25'}
MODEL, TEACHER = '#7454ba', '#287fab'


def csv_rows(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def build(run: Path, parent: Path, previous: dict | None = None) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    summary, rows, decisions, starts, provenance = analyze(run, previous)
    if not rows:
        raise ValueError('No finished attempts yet; active/partial files are deliberately excluded.')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = parent / 'snapshots' / timestamp
    plots = out / 'plots'
    plots.mkdir(parents=True)
    provenance.update(created_at_utc=timestamp, plotting_script_sha256=file_sha256(Path(__file__)))
    lab.write_new(out/'snapshot.json', provenance)
    lab.write_new(out/'summary.json', summary)
    csv_rows(out/'fights.csv', rows)
    csv_rows(out/'decisions.csv', decisions)
    csv_rows(out/'invalid_responses.csv', [d for d in decisions if not d['valid']])
    csv_rows(out/'bank_starts.csv', starts)
    shutil.copy2(__file__, out/'plotting_script_snapshot.py')
    shutil.copy2(ROOT/'src/sts_ai/fight_analysis.py', out/'analysis_module_snapshot.py')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.labelcolor': '#35465d',
        'text.color': '#1b2c43', 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'svg.fonttype': 'none', 'axes.grid': False})
    gallery = []
    completed = [r for r in rows if r['completed']]
    enemies = sorted({e for s in starts for e in s['enemies']})
    n = len(rows)
    pretty = lambda s: s.replace('_', ' ').title()

    def save(fig, slug, title, caption, section='Overview'):
        height=fig.get_figheight()
        fig.suptitle(title, x=.06, y=1-.18/height, ha='left', fontsize=16, fontweight='bold')
        fig.subplots_adjust(left=.10, right=.97, top=1-min(.18,1.05/height),
                            bottom=min(.25,1.35/height), wspace=.35, hspace=.45)
        # Make room for long encounter labels before writing standalone images.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        minimum = min(ax.get_tightbbox(renderer).x0 for ax in fig.axes)
        if minimum < 12:
            fig.subplots_adjust(left=min(.46, .10 + (12-minimum)/fig.bbox.width))
        fig.text(.06, .025, textwrap.fill(caption, 145), fontsize=8, color='#52647b', va='bottom')
        fig.text(.97, 1-.62/height, f'{n:,} finished / {len(starts):,} scheduled', ha='right', fontsize=8, color='#60728a')
        for ext in ('png', 'svg'):
            fig.savefig(plots/f'{slug}.{ext}', dpi=160)
        plt.close(fig)
        gallery.append({'slug': slug, 'title': title, 'caption': caption, 'section': section})

    def single(size=(10, 6)):
        return plt.subplots(figsize=size)

    def empty(ax, message='No observations in this snapshot'):
        ax.text(.5, .5, message, ha='center', va='center', transform=ax.transAxes, color='#77869a')

    def outcomes(ax, groups, *, horizontal=False, normalize=True):
        keys = list(groups)
        positions = np.arange(len(keys))
        offset = np.zeros(len(keys))
        for outcome in COLORS:
            values = np.array([sum(r['outcome'] == outcome for r in groups[k]) for k in keys], dtype=float)
            if normalize:
                values = np.array([v / len(groups[k]) if groups[k] else 0 for k, v in zip(keys, values)])
            if horizontal:
                ax.barh(positions, values, left=offset, label=outcome.title(), color=COLORS[outcome])
            else:
                ax.bar(positions, values, bottom=offset, label=outcome.title(), color=COLORS[outcome])
            offset += values
        labels = [textwrap.fill(f'{k} (n={len(groups[k])})', 38) if horizontal
                  else f'{k} (n={len(groups[k])})' for k in keys]
        if horizontal:
            ax.set_yticks(positions, labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Fraction of finished attempts' if normalize else 'Finished attempts')
        else:
            ax.set_xticks(positions, labels, rotation=30, ha='right')
            ax.set_ylabel('Fraction of finished attempts' if normalize else 'Finished attempts')
        if normalize:
            (ax.set_xlim if horizontal else ax.set_ylim)(0, 1)
        ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(.5, 1.02), ncol=3, frameon=False)

    def paired_scatter(ax, group, x, y, xlabel, ylabel):
        if not group:
            empty(ax)
            return
        for outcome in ('win', 'loss'):
            sub = [r for r in group if r['outcome'] == outcome]
            ax.scatter([r[x] for r in sub], [r[y] for r in sub], s=28, alpha=.65, color=COLORS[outcome], label=outcome.title())
        values = [r[k] for r in group for k in (x, y)]
        low, high = min(values) - 2, max(values) + 2
        ax.plot([low, high], [low, high], '--', color='#718398', lw=1)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(frameon=False, fontsize=8)

    def ecdf(ax, values, label, color):
        if values:
            x = np.sort(values)
            ax.step(x, np.arange(1, len(x)+1)/len(x), where='post', label=label, color=color, lw=2)

    def grouped_box(ax, groups, key):
        valid = [(k, [r[key] for r in group if r.get(key) is not None]) for k, group in groups.items()]
        valid = [(k, v) for k, v in valid if v]
        if not valid:
            empty(ax)
            return
        ax.boxplot([v for k, v in valid], orientation='horizontal', patch_artist=True,
                   boxprops={'facecolor': '#d5caee'}, medianprops={'color': '#513489'})
        ax.set_yticks(range(1, len(valid)+1), [f'{k} (n={len(v)})' for k, v in valid], fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, ls='--', color='#75869a', lw=1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.8))
    axs[0].bar(['Finished', 'Remaining'], [n, len(starts)-n], color=[MODEL, '#cfd9e5'])
    axs[0].set_ylabel('Scheduled attempts')
    for i, value in enumerate((n, len(starts)-n)): axs[0].text(i, value, str(value), ha='center', va='bottom')
    outcomes(axs[1], {'Finished': rows}, normalize=False)
    save(fig, '01_progress', 'Sampling progress and recorded outcomes',
         'Left: finished attempts versus the frozen schedule. Right: wins, deaths and incomplete attempts among finished files. Active fights and archived interruptions are excluded. Incomplete includes invalid responses, errors, timeouts and simulator flags.')

    fig, ax = single()
    cohorts = [rows, completed, rows]
    keys = ['win', 'win', 'invalid']
    labels = ['Win / all finished', 'Win / completed fights', 'Invalid / all finished']
    for i, (cohort, key) in enumerate(zip(cohorts, keys)):
        value = np.mean([r[key] for r in cohort]) if cohort else 0
        interval = cluster_interval(cohort, key)
        ax.bar(i, value, color=[MODEL, COLORS['win'], COLORS['incomplete']][i], width=.6)
        if interval:
            ax.vlines(i, *interval, color='#263a52', lw=2)
            ax.hlines(interval, i-.08, i+.08, color='#263a52')
        ax.text(i, min(value+.045, 1.05), f'{value:.1%}\nn={len(cohort)}', ha='center', fontsize=10)
    ax.set(ylim=(0, 1.15), ylabel='Fraction')
    ax.set_xticks(range(3), labels)
    save(fig, '02_success_rates', 'Success and format validity use different denominators',
         'All-finished win fraction counts incomplete attempts as non-wins; completed-only win fraction excludes them. Lines are descriptive 95% percentile intervals from 2,000 bootstrap samples of whole observed worlds. They are unstable early and are not guarantees for unseen enemies or repeated model samples.')

    order = sorted(enemies, key=lambda e: -sum(e in s['enemies'] for s in starts))
    fig, ax = single((12, 11))
    total = [sum(e in s['enemies'] for s in starts) for e in order]
    observed = [sum(e in r['enemies'] for r in rows) for e in order]
    ax.barh(range(len(order)), total, color='#dbe3ec', label='All scheduled starts')
    ax.barh(range(len(order)), observed, color=MODEL, label='Finished attempts')
    ax.set_yticks(range(len(order)), [pretty(e) for e in order], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel('Fight memberships'); ax.legend(frameon=False)
    for i, (a,b) in enumerate(zip(observed,total)): ax.text(b+2,i,f'{a}/{b}',va='center',fontsize=8)
    ax.set_xlim(0,max(total)*1.2)
    save(fig,'03_enemy_coverage','How much data do we have for each enemy?',
         'Purple shows finished attempts; pale bars show the full scheduled bank. A mixed encounter belongs to every enemy type present, so these bars cannot be added to obtain unique fights. Repeated copies of one enemy in a fight count once. Unsampled enemies remain visible.', 'Coverage')

    fig, ax = single((12, 11))
    outcomes(ax, {pretty(e):[r for r in rows if e in r['enemies']] for e in order}, horizontal=True)
    fig.subplots_adjust(left=.25)
    save(fig,'04_enemy_outcomes','Outcomes by enemy type',
         'Each bar describes all finished attempts containing this enemy; labels give the denominator. Blank bars mean n=0, not a zero win rate. Mixed encounters overlap and small samples can give misleading rankings. Incomplete attempts remain amber.', 'Outcomes')

    def encounter_label(encounter):
        return ' + '.join(pretty(name)+(f' ×{count}' if count>1 else '')
                          for name,count in Counter(encounter.split(' + ')).items())
    groups = {encounter_label(e):[r for r in rows if r['encounter']==e] for e in sorted({r['encounter'] for r in rows})}
    fig, ax = single((14,max(7,len(groups)*.48+2)))
    outcomes(ax, groups, horizontal=True)
    save(fig,'05_encounter_outcomes','Outcomes by exact encounter composition',
         'Unlike enemy membership plots, these groups are mutually exclusive: each fight appears once. Labels retain repeated enemies, so three Sentries is its own composition. Rates are descriptive and labels show sample sizes; empty scheduled encounter types are covered in the coverage table.', 'Outcomes')

    floors = sorted({s['floor'] for s in starts})
    fig, axs = plt.subplots(1,2,figsize=(14,6))
    floor_groups = {str(f):[r for r in rows if r['floor']==f] for f in floors}
    outcomes(axs[0], floor_groups, normalize=False)
    outcomes(axs[1], floor_groups)
    save(fig,'06_floor_outcomes','Outcomes along Act 1',
         'Left: outcome counts at each source floor. Right: the corresponding fractions with all finished attempts in the denominator. Floor is associated with deck growth and encounter difficulty; this is not a causal effect of floor. Floors with no sampled starts have blank bars.', 'Outcomes')

    hp_edges = [0,.25,.5,.75,1.00001]
    hp_labels = ['<25%', '25–<50%', '50–<75%', '75–100%']
    hp_groups = {label:[r for r in rows if lo<=r['hp_fraction']<hi] for label,lo,hi in zip(hp_labels,hp_edges,hp_edges[1:])}
    fig, axs = plt.subplots(1,2,figsize=(12,6))
    outcomes(axs[0],hp_groups,normalize=False); outcomes(axs[1],hp_groups)
    save(fig,'07_entry_hp_outcomes','Outcomes by starting health',
         'Health bands use HP divided by maximum HP at the shared fight start. Counts and rates are shown separately to expose sparse low-health samples. These are source-bank states, not states reached by Gemma in a full run.', 'Outcomes')

    floor_bins = np.arange(.5,max(s['floor'] for s in starts)+1.5)
    health_bins = np.arange(0,max(s['hp'] for s in starts)+11,10)
    def hist(group):
        return np.histogram2d([r['floor'] for r in group],[r['hp'] for r in group],bins=(floor_bins,health_bins))[0].T
    bank_h, observed_h = hist(starts), hist(rows)
    fig, axs = plt.subplots(1,3,figsize=(15,6))
    for ax,h,title,cmap in zip(axs,[bank_h,observed_h,np.divide(observed_h,bank_h,out=np.full_like(bank_h,np.nan),where=bank_h>0)*100],
                              ['Scheduled starts','Finished attempts','Percent sampled'],['Blues','Purples','YlGnBu']):
        mesh=ax.pcolormesh(floor_bins,health_bins,np.ma.masked_invalid(h),cmap=cmap,shading='flat')
        fig.colorbar(mesh,ax=ax,shrink=.8);ax.set(title=title,xlabel='Floor',ylabel='Starting HP')
    save(fig,'08_floor_hp_coverage','Coverage across floor and starting HP',
         'One-floor by 10-HP bins. First two panels are counts with separate colour scales; the last is finished / scheduled percent within each bin. Empty scheduled bins are masked in the percentage panel. This shows which parts of the bank preliminary results represent.', 'Coverage')

    fig, ax = single((12,6))
    wins_h=hist([r for r in rows if r['win']])
    rate=np.divide(wins_h,observed_h,out=np.full_like(wins_h,np.nan),where=observed_h>0)
    mesh=ax.pcolormesh(floor_bins,health_bins,np.ma.masked_invalid(rate),vmin=0,vmax=1,cmap='RdYlGn')
    fig.colorbar(mesh,ax=ax,label='Win / all finished attempts')
    for y,x in np.argwhere(observed_h>0):ax.text(x+1,(health_bins[y]+health_bins[y+1])/2,f'{int(wins_h[y,x])}/{int(observed_h[y,x])}',ha='center',va='center',fontsize=8,color='white' if rate[y,x]>.8 or rate[y,x]<.2 else '#25364b')
    ax.set(xlabel='Floor',ylabel='Starting HP')
    save(fig,'09_floor_hp_success','Win fractions across floor and starting HP',
         'Cells show wins / finished attempts and their ratio as colour. Invalid or failed attempts remain in the denominator. Blank cells have no finished data; a green 1/1 is one observation, not evidence of reliable performance.', 'Outcomes')

    fig, ax = single()
    table=np.zeros((2,2),dtype=int)
    for r in completed:table[int(r['teacher_win']),int(r['win'])]+=1
    ax.imshow(table,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(table[i,j]),ha='center',va='center',fontsize=28,color='white' if table[i,j]>table.max()*.5 else '#25364b')
    ax.set_xticks([0,1],['Gemma loss','Gemma win']);ax.set_yticks([0,1],['Search loss','Search win'])
    save(fig,'10_paired_outcomes','Same starting state: Gemma versus search outcomes',
         f'Only the {len(completed)} model attempts with a resolved fight and no reported simulator flag enter this paired table. The search comparator is the saved source continuation, using privileged native search at every decision. Incomplete model attempts are excluded here and shown in the main outcome plots.', 'Paired comparison')

    fig, ax=single()
    paired_scatter(ax,completed,'teacher_hp_loss','hp_loss','Search net HP loss','Gemma net HP loss')
    save(fig,'11_paired_hp','How much HP does Gemma spend versus search?',
         'Each point is one completed model fight paired with its source search continuation. Above the dashed equality line means Gemma lost more net HP. Deaths are included. Net loss equals entry HP minus post-fight HP and includes relic healing; negative values mean net healing.', 'Paired comparison')

    fig, ax=single()
    gap=[r['hp_loss_gap'] for r in completed]
    if gap:ax.hist(gap,bins=20,color=MODEL,edgecolor='white');ax.axvline(0,color='#3c5068',ls='--')
    else:empty(ax)
    ax.set(xlabel='Gemma net HP loss − search net HP loss',ylabel='Completed paired fights')
    save(fig,'12_hp_gap_distribution','Distribution of the paired HP gap',
         f'Positive values favour search; negative values favour Gemma. Mean gap: {summary["mean_hp_loss_gap"]!s} HP; the exact mean, median and world-bootstrap interval are in summary.json. Ties are retained. This comparison excludes incomplete attempts and includes deaths, so read it beside completion rates.', 'Paired comparison')

    fig, ax=single((13,10))
    grouped_box(ax,{pretty(e):[r for r in completed if e in r['enemies']] for e in order},'hp_loss_gap')
    ax.set_xlabel('Gemma net HP loss − search net HP loss')
    save(fig,'13_enemy_hp_gap','Where is the HP gap largest?',
         'Paired HP-gap distributions by enemy membership. Boxes show quartiles and median; whiskers extend to 1.5×IQR and points outside are retained as outliers. Positive means worse HP preservation by Gemma. Mixed encounters overlap, and n=1 cannot establish a distribution.', 'Paired comparison')

    fig, ax=single((11,8))
    grouped_box(ax,{str(f):[r for r in completed if r['floor']==f] for f in floors},'hp_loss_gap')
    ax.set(xlabel='Gemma net HP loss − search net HP loss',ylabel='Source floor')
    save(fig,'14_floor_hp_gap','Paired HP gap by floor',
         'Only completed paired fights are included; labels give counts. Positive values mean Gemma exits with less HP than search from the identical start. Later-floor differences can reflect encounters and decks rather than floor itself.', 'Paired comparison')

    fig, ax=single()
    paired_scatter(ax,completed,'hp','exit_hp','Starting HP','Gemma post-fight HP')
    save(fig,'15_entry_exit_hp','Starting and final player health',
         'Completed fights only. The dashed line means no net HP change; points above it reflect net healing, often from post-fight relic effects. Red points are deaths at zero HP. Incomplete attempts do not have a final fight HP and are excluded.', 'HP and fight length')

    fig, ax=single()
    ecdf(ax,[r['hp_loss'] for r in completed],'Gemma',MODEL)
    ecdf(ax,[r['teacher_hp_loss'] for r in completed],'Search (same starts)',TEACHER)
    ax.set(xlabel='Net HP loss including post-fight healing',ylabel='Fraction of completed pairs ≤ x');ax.legend(frameon=False)
    save(fig,'16_hp_loss_ecdf','Cumulative HP-loss distributions on matched starts',
         'At a given HP-loss threshold, a higher curve means more fights finished at or below that loss. Both curves use precisely the completed model cohort, so differences are not caused by comparing different starts. These marginal curves complement the paired scatter.', 'Paired comparison')

    fig, ax=single()
    paired_scatter(ax,completed,'teacher_decisions','model_decisions','Search executed decisions','Gemma executed decisions')
    save(fig,'17_decision_counts','Does Gemma need more actions than search?',
         'Each point compares executed combat micro-decisions from the same start. End-turn and potion actions count as decisions. Longer fights can reflect a different strategy or slower wins; deaths can be short, so fewer decisions is not automatically better.', 'HP and fight length')

    fig, ax=single()
    ecdf(ax,[r['turns'] for r in completed],'Gemma',MODEL);ecdf(ax,[r['teacher_turns'] for r in completed],'Search',TEACHER)
    ax.set(xlabel='Last observed decision turn',ylabel='Fraction of completed pairs ≤ x');ax.legend(frameon=False)
    save(fig,'18_turn_counts','Observed combat length in turns',
         'The last player-decision turn in each completed fight, not a count of enemy actions. Both policies use the same completed cohort. This can differ from a game UI final turn counter when an enemy action ends the fight.', 'HP and fight length')

    fig, ax=single()
    token_bins=np.histogram_bin_edges([r['completion_tokens'] for r in rows],bins=20)
    for outcome in COLORS:
        values=[r['completion_tokens'] for r in rows if r['outcome']==outcome]
        if values:ax.hist(values,bins=token_bins,histtype='step',lw=2,label=f'{outcome} (n={len(values)})',color=COLORS[outcome])
    ax.set(xlabel='Generated completion tokens per finished attempt',ylabel='Attempts');ax.legend(frameon=False)
    save(fig,'19_fight_token_cost','Generation cost per fight attempt',
         'Completion-token totals include every saved response, including an invalid final response. Prompt tokens are excluded here. Incomplete attempts can be artificially cheap because they stop early, or expensive if a response reaches its cap.', 'Runtime and tokens')

    fig, ax=single()
    for valid,color,label in [(True,MODEL,'Valid'),(False,COLORS['incomplete'],'Invalid')]:
        values=[max(1,d['completion_tokens']) for d in decisions if d['valid']==valid]
        if values:ax.hist(values,bins=np.geomspace(1,8192,35),histtype='step',lw=2,label=label,color=color)
    ax.axvline(4096,color='#bf3e50',ls='--',label='Response cap');ax.set_xscale('log');ax.set_yscale('symlog',linthresh=1)
    ax.set(xlabel='Completion tokens per response (log scale)',ylabel='Responses (symlog count)');ax.legend(frameon=False)
    save(fig,'20_response_lengths','Short action outputs and long-response tails',
         'Every response from finished attempts is included. The x-axis is logarithmic; y uses a linear region around zero and logarithmic counts above one. The 4,096-token marker highlights cap pressure. Long text can still parse as a valid action under the current last-JSON-object parser.', 'Runtime and tokens')

    fig, ax=single()
    for outcome in COLORS:
        sub=[r for r in rows if r['outcome']==outcome and r['seconds'] is not None]
        ax.scatter([r['completion_tokens'] for r in sub],[r['seconds'] for r in sub],color=COLORS[outcome],label=outcome,alpha=.65)
    ax.set(xlabel='Completion tokens per attempt',ylabel='Attempt wall time (seconds)');ax.legend(frameon=False)
    save(fig,'21_tokens_runtime','How generation length translates into runtime',
         'Attempt wall time includes source replay, inference and result preparation inside the model worker; it excludes initial model loading, pauses and archived interruptions. Colour separates outcomes. This explains throughput but is not a controlled hardware benchmark.', 'Runtime and tokens')

    fig, ax=single()
    ax.scatter([d['prompt_tokens'] for d in decisions],[d['latency_s'] for d in decisions],
               c=[COLORS['incomplete'] if not d['valid'] else MODEL for d in decisions],s=12,alpha=.35)
    ax.set(xlabel='Prompt tokens',ylabel='Decision latency (seconds)')
    save(fig,'22_context_latency','Context size and per-decision latency',
         'Each point is a response from a finished attempt; amber points are invalid. Latency includes prompt processing and generation, so longer output can dominate this relationship. Repeated decisions within a fight are correlated.', 'Runtime and tokens')

    fig, ax=single()
    timed=[r for r in rows if r['seconds'] is not None]
    elapsed=np.cumsum([r['seconds'] for r in timed])/3600
    ax.plot(elapsed,np.arange(1,len(timed)+1),color=MODEL,lw=2)
    ax.set(xlabel='Cumulative recorded attempt time (hours)',ylabel='Finished attempts')
    save(fig,'23_generation_progress','Throughput accumulated over the saved schedule',
         'Timed attempts only, in frozen sampling order. Attempts without recorded durations (such as watchdog timeouts) are omitted from both axes. The x-axis excludes pauses, initial model loads and archived interruptions; it is productive attempt time rather than calendar elapsed time.', 'Runtime and tokens')

    fig, ax=single()
    seconds=[r['seconds'] for r in timed]
    rolling=[np.mean(seconds[max(0,i-19):i+1]) for i in range(len(timed))]
    ax.plot(range(1,len(timed)+1),rolling,color=MODEL)
    ax.set(xlabel='Timed finished attempt in shuffled schedule',ylabel='Mean seconds / attempt (up to last 20)')
    save(fig,'24_rolling_speed','How stable is the runtime estimate?',
         'A trailing mean of up to 20 attempts with recorded durations, in the shuffled order. Untimed timeout/error attempts are omitted, not treated as zero seconds. Encounter mixture, text length and system load affect this measure; it is not a promise about remaining fights.', 'Runtime and tokens')

    fig, ax=single()
    cumulative=np.cumsum([r['win'] for r in rows])/np.arange(1,n+1)
    ax.plot(range(1,n+1),cumulative,label='Win / all finished',color=MODEL)
    denom=np.cumsum([r['completed'] for r in rows]);conditional=np.divide(np.cumsum([r['win'] for r in rows]),denom,out=np.full(n,np.nan),where=denom>0)
    ax.plot(range(1,n+1),conditional,label='Win / completed',color=COLORS['win'])
    ax.set(ylim=(0,1.04),xlabel='Finished attempt in shuffled schedule',ylabel='Cumulative fraction');ax.legend(frameon=False)
    save(fig,'25_cumulative_win_rate','How stable are the preliminary outcome rates?',
         'The two curves differ only in whether incomplete attempts remain in the denominator. They summarize increasing prefixes of a fixed shuffled schedule. They are not a learning curve: model weights are unchanged throughout.', 'Outcomes')

    fig, axs=plt.subplots(1,2,figsize=(12,6))
    invalid=[d for d in decisions if not d['valid']]
    if invalid:
        axs[0].hist([d['decision_index']+1 for d in invalid],bins=min(12,len(invalid)),color=COLORS['incomplete'])
        axs[1].hist([d['completion_tokens'] for d in invalid],bins=min(12,len(invalid)),color=COLORS['incomplete'])
    else:
        for ax in axs:empty(ax,'No invalid responses in this snapshot')
    axs[0].set(xlabel='Response number within fight',ylabel='Invalid responses')
    axs[1].set(xlabel='Tokens in invalid response',ylabel='Invalid responses')
    save(fig,'26_invalid_diagnostics','When do invalid responses stop a trajectory?',
         'Each invalid response is an unexecuted decision that terminates this attempt under the zero-retry protocol. Left shows its position in the fight; right shows response length. The raw outputs and legal start identities are saved in invalid_responses.csv and the gallery audit section.', 'Format and actions')

    fig, ax=single()
    kinds=Counter(d['action_kind'] for d in decisions if d['action_executed'])
    ax.bar(kinds.keys(),kinds.values(),color=MODEL);ax.set_ylabel('Executed actions')
    save(fig,'27_action_mix','What kinds of actions is Gemma executing?',
         'Card types are read from the displayed action text; end turn and potion actions have separate categories. Invalid responses are excluded because no action executed. Other includes selections and actions without a recognized type. This is a descriptive mix, not an optimality score.', 'Format and actions')

    fig, axs=plt.subplots(1,2,figsize=(12,6))
    outcomes(axs[0],{s:[r for r in rows if r['split']==s] for s in ['train','eval']})
    outcomes(axs[1],{s:[r for r in rows if ('ELITE' if any(e in ['GREMLIN_NOB','LAGAVULIN','SENTRY'] for e in r['enemies']) else 'BOSS' if any(e in ['THE_GUARDIAN','HEXAGHOST','SLIME_BOSS'] for e in r['enemies']) else 'NORMAL')==s] for s in ['NORMAL','ELITE','BOSS']})
    save(fig,'28_split_difficulty','Source split and encounter category',
         'Left: source-bank train/eval bookkeeping based on world seed, not evidence that this base model trained on the train split. Right: normal, elite and boss encounter memberships (exclusive for Act 1). All finished attempts are in the denominators; uneven samples limit comparisons.', 'Outcomes')

    enemy_summary=[]
    common_count_max=max(float(hist([s for s in starts if e in s['enemies']]).max()) for e in enemies)
    for enemy in enemies:
        group=[r for r in rows if enemy in r['enemies']]
        bank_group=[s for s in starts if enemy in s['enemies']]
        pairs=[r for r in group if r['completed']]
        enemy_summary.append({'enemy':enemy,'scheduled':len(bank_group),'finished':len(group),
            'wins':sum(r['win'] for r in group),'losses':sum(r['outcome']=='loss' for r in group),
            'incomplete':sum(not r['completed'] for r in group),
            'mean_hp_loss_gap_completed':statistics.mean(r['hp_loss_gap'] for r in pairs) if pairs else None})
        fig,axs=plt.subplots(1,3,figsize=(15,6))
        outcomes(axs[0],{'Sampled':group},normalize=False)
        paired_scatter(axs[1],pairs,'teacher_hp_loss','hp_loss','Search net HP loss','Gemma net HP loss')
        mesh=axs[2].pcolormesh(floor_bins,health_bins,hist(bank_group),cmap='Blues',vmin=0,vmax=common_count_max)
        fig.colorbar(mesh,ax=axs[2],label='Scheduled starts / bin',shrink=.8)
        for outcome in COLORS:
            subset=[r for r in group if r['outcome']==outcome]
            axs[2].scatter([r['floor'] for r in subset],[r['hp'] for r in subset],color=COLORS[outcome],s=22,edgecolor='white',lw=.4)
        axs[2].set(xlabel='Floor',ylabel='Starting HP')
        save(fig,'enemy_'+enemy.lower(),pretty(enemy)+' — encounter profile',
             f'{len(group)} finished attempts from {len(bank_group)} scheduled starts containing {pretty(enemy)}. Left: outcome counts. Middle: paired net HP loss on completed fights; above equality favours search. Right: bank floor/HP density with sampled outcomes overlaid; points can overlap. Density colours share a scale across profiles. Mixed encounters also appear in other enemy profiles.', 'Enemy profiles')
    csv_rows(out/'enemy_summary.csv',enemy_summary)

    # One compact contact sheet is useful for checking every artifact and browsing offline.
    from PIL import Image
    thumbs=[]
    for item in gallery:
        with Image.open(plots/f"{item['slug']}.png") as im:
            thumb=im.convert('RGB');thumb.thumbnail((400,270));thumbs.append(thumb.copy())
    sheet=Image.new('RGB',(1600,((len(thumbs)+3)//4)*290),'#edf1f5')
    for i,thumb in enumerate(thumbs):sheet.paste(thumb,((i%4)*400,(i//4)*290))
    sheet.save(out/'contact_sheet.jpg')
    lab.write_new(out/'captions.json',gallery)
    headers=['Metric','Value']
    stats=[('Finished / scheduled',f'{n} / {len(starts)}'),('Wins / deaths / incomplete',f"{summary['wins']} / {summary['losses']} / {summary['incomplete']}"),
           ('Win / all finished',f"{summary['win_fraction_all_finished']:.1%}"),
           ('Win / completed',f"{summary['win_fraction_completed']:.1%}" if completed else 'n/a'),
           ('Mean paired excess HP loss',f"{summary['mean_hp_loss_gap']:.2f} HP" if completed else 'n/a'),
           ('Worlds observed',str(summary['worlds_observed'])),('Invalid responses / all responses',f"{summary['invalid_responses']} / {summary['responses']}"),
           ('Completion tokens',f"{summary['total_completion_tokens']:,}")]
    limitations='This is an interim snapshot of a shuffled, fixed bank with one model sample per start. Repeated fights from a world are correlated. Enemy panels overlap for mixed encounters. Search is a privileged reference policy. Incomplete model attempts stay in overall success denominators but have no completed-fight HP outcome. Net HP includes post-fight healing. No causal or broad generalization claim follows from these plots. Simulator flags are tracked, but matching saved hashes does not prove simulator correctness.'
    cards=''.join(f'<section class="plot" data-search="{html.escape((g["title"]+" "+g["section"]).lower())}"><h2>{html.escape(g["title"])}</h2><a href="plots/{g["slug"]}.png"><img loading="lazy" src="plots/{g["slug"]}.png" alt="{html.escape(g["title"])}"></a><p>{html.escape(g["caption"])}</p><a href="plots/{g["slug"]}.svg">Download SVG</a></section>' for g in gallery)
    cases=''.join(f'<details><summary>{html.escape(d["state_id"])} · response {d["decision_index"]+1} · {d["completion_tokens"]} tokens</summary><pre>{html.escape(d["raw_response"])}</pre></details>' for d in invalid)
    table='<table>'+''.join(f'<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>' for k,v in stats)+'</table>'
    (out/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Gemma fight analysis</title><style>body{font:16px system-ui;color:#24364b;background:#eef2f6;margin:0}main{max-width:1250px;margin:auto;padding:35px}h1{font-size:34px}p{line-height:1.6}table{border-collapse:collapse;background:white;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #dde4ec}.plot{background:white;padding:24px;margin:28px 0;border-radius:12px}.plot img{width:100%;height:auto}a{color:#455eb5}input{padding:13px;width:90%;font-size:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px}details{margin:12px 0}.note{background:#fff5dc;padding:18px;border-radius:8px}</style></head><body><main><h1>Gemma: single-fight analysis</h1><p>'+html.escape(timestamp)+f' · {len(gallery)} figures, each in PNG and SVG · sampling continues independently.</p>'+table+'<p class="note">'+html.escape(limitations)+'</p><p><a href="summary.json">Summary JSON</a> · <a href="fights.csv">Fight table</a> · <a href="decisions.csv">Decision table</a> · <a href="enemy_summary.csv">Enemy summary</a> · <a href="snapshot.json">Frozen input manifest</a> · <a href="contact_sheet.jpg">All-plot overview</a></p><input id="filter" placeholder="Filter plots: enemy, HP, tokens, outcomes…">'+cards+'<h2>Invalid response audit</h2>'+cases+'<script>document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll(".plot").forEach(x=>x.hidden=!x.dataset.search.includes(q));});</script></main></body></html>')
    caption_text='\n\n'.join(f'## {g["slug"]}: {g["title"]}\n\n{g["caption"]}' for g in gallery)
    (out/'CAPTIONS.md').write_text('# Figure captions\n\n'+limitations+'\n\n'+caption_text+'\n')
    (out/'SUMMARY.md').write_text('# Interim Gemma fight analysis\n\n'+ '\n'.join(f'- {k}: {v}' for k,v in stats)+'\n\n'+limitations+'\n')
    parent.mkdir(parents=True,exist_ok=True)
    relative=out.relative_to(parent)
    temp=parent/'index.tmp'
    temp.write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={relative}/index.html"><a href="{relative}/index.html">Open latest analysis</a>')
    temp.replace(parent/'index.html')
    (parent/'LATEST.txt').write_text(str(out.resolve())+'\n')
    return out


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/gemma_fight_analysis_2026-09-06')
    parser.add_argument('--snapshot',type=Path,help='Rebuild precisely a previous snapshot cohort')
    args=parser.parse_args()
    out=build(args.run.resolve(),args.output.resolve(),lab.read_json(args.snapshot) if args.snapshot else None)
    print(out)


if __name__=='__main__':
    main()
