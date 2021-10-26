from pprint import pformat
import types

from sklearn.metrics import roc_auc_score
import numpy as np

class Accuracy():
    def __init__(self, maximize=True, verbose=False):
        super().__init__(maximize=maximize)
        self.verbose = verbose

    def _test(self, target, approx, mask):
        target = target.reshape(-1).astype(int)
        approx = approx.argmax(2).reshape(-1).astype(int)
        mask = mask.reshape(-1).astype(bool)
        return (target == approx).astype(int)[mask].mean()
    
    def torch(self, approx, target, mask):
        return self._test(target.detach().cpu().numpy(),
                          approx.detach().cpu().numpy(),
                          mask.detach().cpu().numpy())

class AUC():
    def __init__(self, maximize=True, verbose=False):
        super().__init__(maximize=maximize)
        self.verbose = verbose
    
    def _test(self, target, approx, mask):
        scores = []

        label_num = approx[0].shape[-1]
        
        target = target.reshape(-1).astype(int)
        target = np.identity(label_num)[target] # change to one-hot
        approx = approx.reshape([-1, label_num]).astype(float)
        mask = mask.reshape(-1).astype(bool)
        
        target = target[mask]
        approx = approx[mask]

        for i in range(label_num):
            try:
                scores.append(roc_auc_score(target[:, i], approx[:, i]))
            except ValueError:
                pass
            
        return np.mean(scores)

    def torch(self, approx, target, mask):
        return self._test(target.detach().cpu().numpy(),
                          approx.detach().cpu().numpy(), 
                          mask.detach().cpu().numpy())