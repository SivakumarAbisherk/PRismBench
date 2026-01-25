import json
import sys
from collections import Counter
import pandas as pd


def analyze_labeling_quality(accepted_df: pd.DataFrame, human_df: pd.DataFrame):
    print('='*80)
    print('LABELING QUALITY ANALYSIS')
    print('='*80)

    total = len(accepted_df) + len(human_df)

    print(f'\n📊 Overall Statistics:')
    print(f'  Total PRs processed: {total}')
    print(f'  ✅ Accepted (consensus): {len(accepted_df)} ({len(accepted_df)/total*100:.1f}%)')
    print(f'  ⚠️  Need human review: {len(human_df)} ({len(human_df)/total*100:.1f}%)')

    if len(accepted_df) > 0:
        print(f'\n🎯 Consensus Breakdown:')
        vote_counts = accepted_df['vote_count'].value_counts().sort_index(ascending=False)
        for votes, count in vote_counts.items():
            print(f'  {votes}/3 models agreed: {count} PRs ({count/len(accepted_df)*100:.1f}%)')

        print(f'\n🏷️  Label Distribution (Accepted PRs):')
        all_labels = []
        for labels_str in accepted_df['agreed_labels']:
            if labels_str:
                all_labels.extend(labels_str.split(';'))

        label_counts = Counter(all_labels)
        for label, count in label_counts.most_common():
            print(f'  {label}: {count} ({count/len(accepted_df)*100:.1f}%)')

        print(f'\n📈 Confidence Analysis:')
        confidences = []
        for exp_str in accepted_df['explanations']:
            try:
                exp_dict = json.loads(exp_str)
                for label, details in exp_dict.items():
                    if isinstance(details, dict) and 'confidence' in details:
                        confidences.append(details['confidence'])
            except:
                pass

        if confidences:
            import numpy as np
            print(f'  Average confidence: {np.mean(confidences):.3f}')
            print(f'  Median confidence: {np.median(confidences):.3f}')
            print(f'  Min confidence: {np.min(confidences):.3f}')
            print(f'  Max confidence: {np.max(confidences):.3f}')

            low_conf_threshold = 0.7
            low_conf_count = sum(1 for c in confidences if c < low_conf_threshold)
            if low_conf_count > 0:
                print(f'  ⚠️  Low confidence (<{low_conf_threshold}): {low_conf_count} labels')

    if len(human_df) > 0:
        print(f'\n❓ Ambiguous Cases Needing Human Review:')
        print(f'  Total: {len(human_df)}')
        print(f'\n  Sample reasons for human review:')
        for idx, row in human_df.head(5).iterrows():
            pr_num = row['pr_number']
            try:
                responses = json.loads(row['model_responses_summary'])
                label_sets = [set(r.get('labels', [])) for r in responses if not r.get('error')]
                print(f'  PR #{pr_num}: {len(label_sets)} models responded with {len(set(tuple(sorted(s)) for s in label_sets))} different label sets')
            except:
                print(f'  PR #{pr_num}: Error parsing responses')

    print('\n' + '='*80)

    print('\n💡 Recommendations:')
    if total > 0 and len(accepted_df) / total < 0.5:
        print('  ⚠️  Low consensus rate (<50%). Consider:')
        print('     - Reviewing system instructions for clarity')
        print('     - Checking if prompt construction is correct')
        print('     - Lowering consensus threshold to 1/3 (with caution)')

    if len(accepted_df) > 0 and 'confidences' in locals() and confidences:
        import numpy as np
        avg_conf = np.mean(confidences)
        if avg_conf < 0.75:
            print('  ⚠️  Low average confidence. Consider:')
            print('     - Adding more context to prompts')
            print('     - Using higher-quality models')
            print('     - Requiring unanimous (3/3) consensus')

    print('\n✅ Next steps:')
    print('  1. Manually review a random sample of accepted labels')
    print('  2. Process human_needed.csv through manual labeling')
    print('  3. Compare LLM labels vs human labels on overlap')
    print('  4. Adjust CONSENSUS_THRESHOLD based on precision/recall needs')

    return {
        'total': total,
        'accepted': len(accepted_df),
        'human_needed': len(human_df),
        'consensus_rate': len(accepted_df) / total if total > 0 else 0,
        'label_distribution': dict(label_counts) if len(accepted_df) > 0 else {},
        'avg_confidence': np.mean(confidences) if 'confidences' in locals() and confidences else None
    }
