

from sklearn.base import TransformerMixin


class ValMasker(TransformerMixin):
    """ Mask values in a DataFrame according to values in a column already
    present in the data by setting them to a specific value. """

    def __init__(self, val=0., mask=None, mask_var='landsea', mask_val=1):
        self.mask = mask
        self.mask_var = mask_var
        self.mask_val = mask_val
        self.val = val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _X = X.copy()
        if self.mask is not None:
            _X[self.mask] = self.val
        else:
            _X[_X[self.mask_var] == self.mask_val] = self.val
        return _X
