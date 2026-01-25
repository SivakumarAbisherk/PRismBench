import os
import time
import json
import traceback
from typing import Dict, List, Callable
import pandas as pd
from tqdm import tqdm

from .config import (
    MODEL_OUTPUTS_BASE_DIR,
    PER_MODEL_FILE_PATTERN,
    LLM_SLEEP
)
from .prompt_builder import build_prompt
from .response_handler import validate_llm_response


def process_pr_with_single_model(
    pr: Dict, 
    model_name: str, 
    query_function: Callable
) -> Dict:
    pr_number = pr.get('pr_number', 'unknown')
    
    try:
        prompt = build_prompt(pr)
        response = query_function(prompt, pr)
        response = validate_llm_response(response, model_name, pr_number)
        if 'explanations' in response and isinstance(response['explanations'], list):
            for exp in response['explanations']:
                if 'confidence' in exp and isinstance(exp['confidence'], str):
                    try:
                        exp['confidence'] = float(exp['confidence'])
                    except:
                        exp['confidence'] = 0.5

        return {
            'pr_number': pr_number,
            'model': model_name,
            'timestamp': time.time(),
            'prompt': prompt,
            'risk_type_labels': response.get('risk_type_labels', []),
            'explanations': response.get('explanations', []),
            'raw_response': json.dumps(response),
            'error': None,
            'success': True
        }
    except Exception as e:
        print(f'ERROR: {model_name} failed for PR {pr_number}: {e}')
        traceback.print_exc()
        return {
            'pr_number': pr_number,
            'model': model_name,
            'timestamp': time.time(),
            'prompt': '',
            'risk_type_labels': [],
            'explanations': [],
            'raw_response': '',
            'error': str(e),
            'success': False
        }


def process_all_prs_with_model(
    pr_list: List[Dict],
    model_name: str,
    query_function: Callable,
    loop_number: int
) -> pd.DataFrame:
    print('\n' + '='*80)
    print(f'🤖 Processing {len(pr_list)} PRs with {model_name}...')
    print('='*80)

    results = []
    for idx, pr in enumerate(tqdm(pr_list, desc=f'{model_name} progress')):
        result = process_pr_with_single_model(pr, model_name, query_function)
        results.append(result)
        if idx < len(pr_list) - 1:
            time.sleep(LLM_SLEEP)
    df = pd.DataFrame(results)
    output_file = PER_MODEL_FILE_PATTERN.format(
        model_name=model_name, 
        loop_number=loop_number
    )
    output_path = os.path.join(MODEL_OUTPUTS_BASE_DIR, model_name.lower(), output_file)
    df.to_csv(output_path, index=False)
    print(f'✅ Saved {model_name} results to {output_path}')
    success_count = df['success'].sum() if 'success' in df.columns else 0
    fail_count = len(df) - success_count
    print(f'   Success: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)')
    if fail_count > 0:
        print(f'   Failures: {fail_count}')

    return df
