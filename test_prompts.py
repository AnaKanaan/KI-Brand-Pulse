import json, os

def test_prompts_exist():
 with open('coder_prompts_passB.json','r',encoding='utf-8') as f:
  d=json.load(f)
 assert 'system_coder' in d and 'user_coder' in d
 for lang in ['de','fr','it','rm','en']:
  assert lang in d['user_coder']
