from typing import List, Dict, Tuple
from .config import CONSENSUS_THRESHOLD
from .response_handler import normalize_labels


def responses_agree(
    responses: List[Dict], 
    threshold: int = CONSENSUS_THRESHOLD
) -> Tuple[bool, Tuple[str], List[Dict]]:
    valid_responses = [r for r in responses if not r.get('error')]
    
    if len(valid_responses) < threshold:
        return (False, tuple(), [])
    normalized = [
        normalize_labels(r.get('risk_type_labels') or r.get('labels') or []) 
        for r in valid_responses
    ]
    counts = {}
    response_map = {}
    for i, lbl_tuple in enumerate(normalized):
        counts[lbl_tuple] = counts.get(lbl_tuple, 0) + 1
        response_map.setdefault(lbl_tuple, []).append(valid_responses[i])

    if not counts:
        return (False, tuple(), [])
    best_key, best_count = max(counts.items(), key=lambda kv: kv[1])

    if not best_key:
        return (False, tuple(), response_map.get(best_key, []))
    agree = best_count >= threshold
    winning_explanations = response_map.get(best_key, [])

    return (agree, best_key, winning_explanations)
