0. higher learning rate leads to osolcating learning curve
1. hihger batch size is slower in training while can make training more effective - val_loss descends faster

# Ablation study
1. rmsnorm: prenorm is better than postnorm is much better than no norm:
   1. baseline-batch-128_20260821_050222
   2. baseline-post-rmsnorm2_20260822_003608
   3. baseline-no-rmsnorm_20260821_235159
2. rope: rope > no-rope
   1. baseline-batch-128_20260821_050222 vs baseline-no-rope_20260823_231132
3. swiglu vs silu: swiglu>silu
   1.  baseline-batch-128_20260821_050222 vs baseline-silu_20260823_231912