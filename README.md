# Bitdot-attention
## Requirements
```
torch>=2.7
```
## Files
```
test.py: prompt, design, correct and readable implementation
fastflex.py: faster implementation. Correct when max_chunk_bits < 30. We use 28 here.
diff_fastflex.py: modify max_chunk_bits and see diff results
```
Attention flag compression optimize is TBD.