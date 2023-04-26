# multi-label intent classification

## multi-label intent classification on ./nlupp/banking

```
python multi-label_nlupp_banking.py --b {b} --e {e} --lr {lr} --regime {regime} --fold {fold}
```
### Arguments
* b: batch size (default=8)
* e: epoch (default=50)
* lr: learning rate (default=1e-4)
* regime: {low, mid, large} (default=mid)
* fold: (default=None) (use fold=1 or fold=2 to debug)

### Examples

```
python multi-label_nlupp_banking.py --b 8 --e 50 --lr 1e-4 --regime low # --fold 20
python multi-label_nlupp_banking.py --b 8 --e 50 --lr 1e-4 --regime mid # --fold 10
python multi-label_nlupp_banking.py --b 8 --e 30 --lr 2e-5 --regime large # --fold 10
```

## multi-class intent classification on ./banking

```
python multi-class_banking.py
```

