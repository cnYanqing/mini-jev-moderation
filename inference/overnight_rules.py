"""Small shared guards for dataset isolation and conservative model selection."""
def validate_partition(rows):
    ids=set();pixels=set()
    for row in rows:
        if row['id'] in ids or row['pixel'] in pixels:raise ValueError('Duplicate sample or normalized pixels')
        if row['split'] not in ('train','dev','test'):raise ValueError('Unknown split')
        ids.add(row['id']);pixels.add(row['pixel'])

def select_candidate(candidates):
    baseline=candidates[0]
    eligible=[c for c in candidates if c['fn']<=baseline['fn'] and c['fp']<=baseline['fp']]
    return min(eligible,key=lambda c:(c['fn']+c['fp'],c['brier'],c['epoch'],str(c.get('trial','parent'))))

def decision(sexual,gore,block_thresholds,review_threshold,diagram_score=None,diagram_threshold=1.01,monochrome=False):
    import math
    if any(not math.isfinite(v) or not 0<=v<=1 for v in (sexual,gore)):raise ValueError('Invalid score')
    rt=review_threshold if isinstance(review_threshold,dict) else {'sexual':review_threshold,'gore':review_threshold}
    for label in ('sexual','gore'):
        if not math.isfinite(block_thresholds[label]) or not 0<=rt[label]<=block_thresholds[label]<=1.01:raise ValueError('Invalid thresholds')
    if diagram_score is not None and (not math.isfinite(diagram_score) or not 0<=diagram_score<=1 or not 0<=diagram_threshold<=1):raise ValueError('Invalid diagram score/threshold')
    if sexual>=block_thresholds['sexual']:return 'block'
    if gore>=block_thresholds['gore']:
        return 'review' if monochrome or diagram_score is not None and diagram_score>=diagram_threshold else 'block'
    if sexual>=rt['sexual'] or gore>=rt['gore']:return 'review'
    return 'allow'

def runtime_mode(release,device):
    return release['mode'] if device==release['verified_device'] else 'review_only'

def apply_mode(action,mode):
    return action if mode=='automatic_pilot' else 'review'

def runtime_versions():
    from importlib.metadata import version
    return {name:version(name) for name in ('torch','transformers','peft','numpy','pillow')}
