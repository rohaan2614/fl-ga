# Experiments on RIT Computing Cluster

Rounds => 1e4

| Job ID | Settings | Acc | 
|--------|----------|-------|
| 19263606 | batch size => 256 | **74.9%**| 
| 19276919 | batch size 256 -> 128 | 64.2% |
| 19276920 | batch size 128 -> 384 | 72.1% |
| 19276921 | batch size 128 -> 512 | 71.1% |


* Batch Size => 256
* Rounds => 1e4

| Job ID | Settings | Acc | 
|--------|----------|-------|
| 19263606 | lr => 0.01 | 74.9% |
| 19279403 | lr 0.01 -> 0.005 | 64.3% |
| 19279402 | lr 0.01 -> 0.001 | 56.8% |
| 19279794 | lr 0.01 -> 0.11 | 80.6% |
| 19279404 | lr 0.01 -> 0.1 | **81.9%** |
| 19280848 | lr 0.01 -> 0.09 | **81.9%** |
| 19280943 | lr 0.01 -> 0.08 | 81.7% |
| 19280944 | lr 0.01 -> 0.06 | 81.1% |
| 19279596 | lr 0.1 -> 0.2 | 0.1% |
| 19279604 | lr 0.1 -> 0.5 | 0.1% |
| 19279607 | lr 0.1 -> 1 | 0.1% |

* Batch Size => 256
* Rounds => 1e4
* LR => 0.1

| Job ID | Settings | Acc | 
|--------|----------|-------|
| 19279404 | momentum => 0.9 | **81.9%** |
| 19280949 | momentum 0.9 -> 0.99 | ?% |
| 19280946 | momentum 0.9 -> 0.95 | 81.3% |
| 19280947 | momentum 0.9 -> 0.85 | 80.2% |
| 19280948 | momentum 0.9 -> 0.5 | 72.6% |
| 19280951 | momentum 0.9 -> 0.25 | ?% |

* Batch Size => 256
* Rounds => 1e4
* LR => 0.1
* momentum => 0.9

| Job ID | Settings | Acc | 
|--------|----------|-------|
| 19279404 | weight decay => 0 | **81.9%** |
| 19281020 | weight decay 0 => 1e-2 | 80.8% |
| 19281021 | weight decay 0 => 1e-3 | 80.6% |
| 19281022 | weight decay 0 => 1e-4 | 81.6% |
| 19281023 | weight decay 0 => 1e-5 | 81.2% |
| 19281024 | weight decay 0 => 1e-6 | 80.4% |

| Job ID | Settings | Acc | 
|--------|----------|-------|
| 19320082 | weight decay => 0, nesterov = True | 80.3% |
| 19327882 | weight decay 0 => 1e-1, nesterov = True | ?% |
| 19320875 | weight decay 0 => 1e-2, nesterov = True | ?% |
| 19320651 | weight decay 0 => 1e-3, nesterov = True | ?% |
| 19320418 | weight decay 0 => 1e-4, nesterov = True | ?% |
| 19309871 | weight decay 0 => 1e-5, nesterov = True | 78.2% |
| 19309731 | weight decay 0 => 1e-6, nesterov = True | **82.2%** |
| 19320670 | weight decay 0 => 1e-7, nesterov = True | ?% |
| 19320430 | weight decay 0 => 1e-8, nesterov = True | ?% |

* Batch Size => 256
* Rounds => 1e4
* LR => 0.1
* momentum => 0.9
* weight decay => 0.1
* nesterov => True