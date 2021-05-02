import xgboost as xgb


class XGB(xgb.XGBClassifier):
    def __init__(self, params=None):
        r_state = {'random_state': 1}
        if params:
            params.update(r_state)
        else:
            params = r_state
        super(XGB, self).__init__(**params)

    def fit(self, tr_x, tr_y, va_x=None, va_y=None, verbose=True, early_stopping_rounds=None,
            sample_weight=None, base_margin=None, eval_set=None, eval_metric=None, xgb_model=None,
            sample_weight_eval_set=None, base_margin_eval_set=None, feature_weights=None, callbacks=None):
        eval_metric = 'mlogloss'
        eval_set = [(tr_x, tr_y), (va_x, va_y)]
        super(XGB, self).fit(tr_x, tr_y, va_x, va_y,
                             eval_metric=eval_metric,
                             eval_set=eval_set,
                             verbose=verbose,
                             early_stopping_rounds=early_stopping_rounds)
