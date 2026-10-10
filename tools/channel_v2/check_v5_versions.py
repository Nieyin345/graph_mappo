
for p in ['.archive/historical_paper_drafts/论文初稿_中文_v5_20261003.md',
          '.archive/historical_paper_drafts/论文初稿_中文_v5_前三章审阅润色_20261004.md',
          '.archive/historical_paper_drafts/论文初稿_中文_v5.3_I-III优化_20261004.md']:
    with open(p, 'r', encoding='utf-8') as f:
        first_lines = [next(f) for _ in range(15)]
    print(f'=== {p} ===')
    print(''.join(first_lines))
