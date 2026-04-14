# Corpus Quality Review

## L1_synthetic
- docs: 700,000
- chars: 128.6 MB  (~32,160,492 tokens est.)
- doc length: median=176, p90=279, max=397
- inline math `$...$` occurrences: 0  (0.0/doc)
- display math `$$...$$` occurrences: 0
- by level: {1: 700000}
- by type/subtype (top 8): {'addition': 105199, 'multiplication': 90950, 'subtraction': 84153, 'division': 70131, 'order_ops': 55783, 'frac_add': 55719, 'percent': 49253, 'frac_mul': 49132}

## openstax
- docs: 797
- chars: 17.3 MB  (~4,324,268 tokens est.)
- doc length: median=22382, p90=37424, max=128113
- inline math `$...$` occurrences: 176,089  (220.9/doc)
- display math `$$...$$` occurrences: 10,124
- by level: {2: 496, 4: 163, 3: 138}
- by type/subtype (top 8): {'?': 797}

## aim
- docs: 120
- chars: 2.9 MB  (~716,749 tokens est.)
- doc length: median=22163, p90=48528, max=74991
- inline math `$...$` occurrences: 30,729  (256.1/doc)
- display math `$$...$$` occurrences: 1,365
- by level: {5: 95, 6: 25}
- by type/subtype (top 8): {'?': 120}

## proofwiki
- docs: 36,899
- chars: 16.2 MB  (~4,061,818 tokens est.)
- doc length: median=334, p90=760, max=20468
- inline math `$...$` occurrences: 291,346  (7.9/doc)
- display math `$$...$$` occurrences: 9
- by level: {7: 23134, 5: 13765}
- by type/subtype (top 8): {'theorem': 16369, 'definition': 13562, 'proof': 6765, 'axiom': 203}

## Grand Total
- docs: 737,816
- chars: 165.1 MB
- tokens (char/4 est.): ~41.3M