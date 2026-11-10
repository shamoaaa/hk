import numpy as np
import pandas as pd
import xgboost as xgb

from .breeding_utils import generate_breeding_features_v1
from .oestrus_utils import generate_oestrus_features_v1
from .predict_utils import get_perpig_predout


def is_oestrus_base_oestrus_model(model, feature_df, threshold=0.35):
    type = 'list'
    extract_feature, pig_pred_mask, pig_dates = generate_oestrus_features_v1([('pig_id', feature_df)])

    if pig_pred_mask[-1] == -1:
        return 0, 0

    xgb_input = xgb.DMatrix(extract_feature)
    pred = model.predict(xgb_input)
    perpig_pred = get_perpig_predout(pred, pig_pred_mask, pig_dates, threshold, type)

    return abs(pred[0]) if abs(pred[0])<1 else 1.0, perpig_pred[-1]


def is_oestrus_base_breeding_model(model, feature_df, threshold=0.3):
    type = 'list'
    extract_feature, pig_pred_mask, pig_dates = generate_breeding_features_v1([('pig_id', feature_df)])

    if pig_pred_mask[-1] == -1:
        return 0, 0

    xgb_input = xgb.DMatrix(extract_feature)
    pred = model.predict(xgb_input)
    perpig_pred = get_perpig_predout(pred, pig_pred_mask, pig_dates, threshold, type)

    return abs(pred[0]) if abs(pred[0])<1 else 1.0, perpig_pred[-1]


def is_oestrus_base_breeding_model_v2(model, feature_df, threshold=0.304):
    """
    配种检测模型 v2（youhua 3天版90维 + 重心选日）。
    逻辑结构与 is_oestrus_base_breeding_model 完全一致：
      generate_breeding_features_v2 → 模型预测 → get_perpig_predout_v2
    """
    from .breeding_utils import generate_breeding_features_v2
    from .predict_utils import get_perpig_predout_v2

    type = 'list'
    extract_feature, pig_pred_mask, pig_dates = generate_breeding_features_v2([('pig_id', feature_df)])

    if pig_pred_mask[-1] == -1:
        return 0, 0

    xgb_input = xgb.DMatrix(extract_feature)
    pred = model.predict(xgb_input)
    perpig_pred = get_perpig_predout_v2(pred, pig_pred_mask, pig_dates, threshold, type)

    return abs(pred[0]) if abs(pred[0]) < 1 else 1.0, perpig_pred[-1]


def generate_input(act_feature, br_feature, is_up_feature, ideal_channel_feature):
    act_feature = np.array(act_feature[-3:]).flatten()
    br_feature = np.array(br_feature[-3:]).flatten()
    is_up_feature = np.array(is_up_feature[-3:]).flatten()

    df = pd.DataFrame(list(zip(act_feature, br_feature, is_up_feature)), columns=['normalize_act', 'br', 'is_up'])
    df[df < 0] = np.nan

    return df
