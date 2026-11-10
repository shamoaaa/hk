import numpy as np

def get_perpig_predout(pred, pig_pred_mask, pig_date, threshold, type='list'):
    '''
    pred: 数组，xgb模型预测输出概率值
    pig_pred_mask: 数组，每头猪被预测的天的掩码 -1 表示该天特征不足模型不预测
    pig_date: 字典，每头猪的输入数据天数，按一天2880条数据计算 
    Return:
    根据type选择返回形式
    'zip':
    字典形式的猪按天预测结果
    {pig_id:[-1,-1,0,1,1],...} -1表示特征不足模型拒绝预测,0表示不配种输出,1表示配种输出
    'list':
    单个数组表示输入每头猪的最新一天预测结果 [-1,0,1]
    结果含义：
    -1 - 特征不足模型拒绝预测
    0 - 不配种
    1 - 需要配种
    '''
    pred = [p>threshold for p in pred] # xgb1.0 阈值0.5，可调整
    pred = np.array(pred)
    pig_pred_mask = np.array(pig_pred_mask)
    indices = np.where(pig_pred_mask==1)[0] # 获取模型预测的天位置

    # assert(len(indices)==len(pred)), f"待预测天数和模型预测天数不匹配!待预测天数:{len(indices)},模型预测天数:{len(pred)}" # 确保待预测天数和模型预测天数一致

    pig_pred_mask[indices]=pred
    perpig_pred = {}
    list_out = []
    start = 0

    for pig in pig_date:
        date = pig_date[pig]
        p = pig_pred_mask[start:start+date]
        list_out.append(p[-1])
        perpig_pred[pig] = p
        start += date
        
    if type=='dict':
        return perpig_pred
    if type=='list':
        return list_out



def get_perpig_predout_v2(pred, pig_pred_mask, pig_date, threshold, type='list'):
    '''
    V2 后处理：逐天阈值判断（与生产 get_pred 逻辑兼容）。
    
    与 get_perpig_predout 完全一致的逻辑：
      - 逐天概率 > threshold → 报1，否则0
      - 返回每头猪最新一天的预测结果
    
    生产环境中，每天只传入3天数据、只产出1天预测，因此后处理必须是
    "逐天独立决策"——不能做重心选日（需要看未来数据）。
    配种延后确认/数据库写入等逻辑由 ezviz_backend.get_pred 负责，不在此处。
    
    返回:
    'list': 单个数组表示输入每头猪的最新一天预测结果 [-1, 0, 1]
      -1: 特征不足模型拒绝预测
       0: 不配种
       1: 需要配种
    '''
    pred = [p > threshold for p in pred]
    pred = np.array(pred)
    pig_pred_mask = np.array(pig_pred_mask)
    indices = np.where(pig_pred_mask == 1)[0]

    pig_pred_mask[indices] = pred
    list_out = []
    start = 0

    for pig in pig_date:
        date = pig_date[pig]
        p = pig_pred_mask[start:start + date]
        list_out.append(p[-1])
        start += date

    if type == 'list':
        return list_out
