import numpy as np
from ms_common.log import logger


def is_oestrus_base_logic(act_feature, br_feature, is_up_feature, ideal_channel_feature):
    br_bias = 1

    # is_oestrus, is_breeding, confidence = self.is_oestrus_base_act(act_feature, today)
    is_oestrus, is_breeding, confidence = is_oestrus_base_up(is_up_feature)
    base_br = is_oestrus_base_br(br_feature, br_bias)

    return is_oestrus, is_breeding, base_br, confidence


def is_oestrus_base_logic_jingchan(act_feature, br_feature, is_up_feature, ideal_channel_feature):
    br_bias = 1

    # is_oestrus, is_breeding, confidence = self.is_oestrus_base_act(act_feature, today)
    is_oestrus, is_breeding, confidence = is_oestrus_base_up_jingchan(is_up_feature)
    # base_br = is_oestrus_base_br(br_feature, br_bias)

    return is_oestrus, is_breeding, 0, confidence


def is_oestrus_base_br(br_feature, br_bias):
    if len(br_feature) < br_bias:
        logger.error('br_num is zero! ')
        return False
    else:

        need_br_feature = br_feature[-br_bias-1][0:7*120]
        over_40_minutes = 0

        for main_freq in need_br_feature:
            if main_freq >= 40:
                over_40_minutes += 1

        if over_40_minutes >= 120:
            return True
        else:
            return False


def is_oestrus_base_act(act_feature):
    is_oestrus = 0
    is_breeding = 0
    confidence = 0

    features = []
    minutes = np.linspace(0, len(act_feature[0]), 60 * 8 + 1, dtype=int)

    for act in act_feature:
        feature = 0  # 0-8点活动的分钟数
        # 遍历所有时间
        for minute in range(60 * 8):
            if np.mean(act[minutes[minute]:minutes[minute + 1]]) > 3:
                feature += 1

        features.append(feature)

    if len(features) <= 1:
        return is_oestrus, is_breeding, confidence
    else:
        today_feature = features[-1]
        yesterday_feature = features[-2]
        if yesterday_feature == 0:
            if today_feature >= 30:
                is_oestrus = 1
                is_breeding = 2
                confidence = 0.7
        else:
            times = today_feature / yesterday_feature

            # ---------- 发情 ----------
            if ((yesterday_feature == 1 and times >= 20) or
                    (yesterday_feature == 2 and times >= 15) or
                    (yesterday_feature == 3 and times >= 10) or
                    (yesterday_feature == 4 and times >= 5) or
                    (yesterday_feature == 5 and times >= 4) or
                    (yesterday_feature == 6 and times >= 3) or
                    (15 >= yesterday_feature >= 7 and times >= 2) or
                    (20 > yesterday_feature > 15 and times >= 1.5) or
                    (yesterday_feature >= 20 and times >= 1.25)):
                is_oestrus = 1
                confidence = 0.8

                if ((times >= 3 and yesterday_feature >= 10 and today_feature <= 300) or
                        (times >= 5 and today_feature <= 300)):
                    confidence = 0.9

            # ---------- 配种 ----------
            if ((yesterday_feature == 1 and times >= 20) or
                    (yesterday_feature == 2 and times >= 15) or
                    (yesterday_feature == 3 and times >= 10) or
                    (yesterday_feature == 4 and times >= 5) or
                    (yesterday_feature == 5 and times >= 4) or
                    (yesterday_feature == 6 and times >= 3) or
                    (15 >= yesterday_feature >= 7 and times >= 2) or
                    (yesterday_feature > 15 and times >= 1.5)):
                is_breeding = 2
                confidence = 0.8

                if ((times >= 3 and yesterday_feature >= 10 and today_feature <= 300) or
                        (times >= 5 and today_feature <= 300)):
                    confidence = 0.9

            if (times >= 3 and yesterday_feature >= 10 and today_feature <= 300) or (
                    times >= 5 and today_feature <= 300):
                confidence = 0.9

        # ---------- 发情 ----------
        if len(features) >= 3:
            before_yesterday_feature = features[-3]

            if today_feature >= yesterday_feature >= before_yesterday_feature and today_feature != 0:
                is_oestrus = 1
                confidence = max(confidence, 0.8)

        # ---------- 配种 ----------
        if len(features) >= 4:
            before_yesterday_feature = features[-3]
            before_before_yesterday_feature = features[-4]

            if today_feature >= yesterday_feature >= before_yesterday_feature >= before_before_yesterday_feature and today_feature != 0:
                is_breeding = 2
                confidence = max(confidence, 0.8)

                if dizeng_beishu(before_before_yesterday_feature, before_yesterday_feature) or dizeng_beishu(
                        before_yesterday_feature, yesterday_feature) or dizeng_beishu(yesterday_feature,
                                                                                      today_feature):
                    confidence = 1

        return is_oestrus, is_breeding, confidence


def is_oestrus_base_up(is_up_feature):
    is_oestrus = 0
    is_breeding = 0
    confidence = 0

    features = []

    for one_day_is_up in is_up_feature:
        feature = 0
        for i in one_day_is_up[0: 7*60*2]:
            if i == 1:
                feature += 1
        features.append(feature)

    if len(features) <= 1:
        return is_oestrus, is_breeding, confidence
    else:
        today_feature = features[-1]
        yesterday_feature = features[-2]
        if yesterday_feature == 0:
            if today_feature >= 30:
                is_oestrus = 1
                is_breeding = 2
                confidence = 0.7
        else:
            times = today_feature / yesterday_feature
            diff = today_feature - yesterday_feature

            # ---------- 发情、配种 ----------
            if ((yesterday_feature == 0 and times >= 30) or
                    (yesterday_feature == 1 and times >= 40) or
                    (yesterday_feature == 2 and times >= 20) or
                    (yesterday_feature == 3 and times >= 15) or
                    (yesterday_feature == 4 and times >= 10) or
                    (yesterday_feature == 5 and times >= 8) or
                    (yesterday_feature == 6 and times >= 6) or
                    (10 >= yesterday_feature >= 7 and times >= 4) or
                    (yesterday_feature > 10 and times >= 2.5) or
                    (yesterday_feature > 25 and times >= 2) or
                    (yesterday_feature > 35 and times >= 1.5) or
                    (yesterday_feature > 50 and diff >= 25)):
                is_oestrus = 1
                is_breeding = 2
                confidence = 0.8

        # ---------- 发情 ----------
        if len(features) >= 3:
            before_yesterday_feature = features[-3]

            if today_feature >= yesterday_feature >= before_yesterday_feature and today_feature != 0:
                is_oestrus = 1
                is_breeding = 2
                confidence = max(confidence, 0.8)
                if dizeng_beishu(before_yesterday_feature, yesterday_feature) or dizeng_beishu(yesterday_feature,
                                                                                               today_feature):
                    confidence = 1

        # # ---------- 配种 ----------
        # if len(features) >= 4:
        #     before_yesterday_feature = features[-3]
        #     before_before_yesterday_feature = features[-4]
        #
        #     if today_feature >= yesterday_feature >= before_yesterday_feature >= before_before_yesterday_feature and today_feature != 0:
        #         is_breeding = 2
        #         confidence = max(confidence, 0.8)
        #
        #         if dizeng_beishu(before_before_yesterday_feature, before_yesterday_feature) or dizeng_beishu(
        #                 before_yesterday_feature, yesterday_feature) or dizeng_beishu(yesterday_feature,
        #                                                                               today_feature):
        #             confidence = 1

        return is_oestrus, is_breeding, confidence


def is_oestrus_base_up_jingchan(is_up_feature):
    is_oestrus = 0
    is_breeding = 0
    confidence = 0

    features = []

    for one_day_is_up in is_up_feature:
        feature = 0
        for i in one_day_is_up[0: 7*60*2]:
            if i == 1:
                feature += 1
        features.append(feature)

    if len(features) <= 1:
        return is_oestrus, is_breeding, confidence
    else:
        today_feature = features[-1]
        yesterday_feature = features[-2]
        if yesterday_feature == 0:
            if today_feature >= 30:
                is_oestrus = 1
                is_breeding = 2
                confidence = 0.7
        else:
            times = today_feature / yesterday_feature
            diff = today_feature - yesterday_feature

            # ---------- 发情、配种 ----------
            if ((yesterday_feature == 0 and times >= 30) or
                    (yesterday_feature == 1 and times >= 40) or
                    (yesterday_feature == 2 and times >= 20) or
                    (yesterday_feature == 3 and times >= 15) or
                    (yesterday_feature == 4 and times >= 10) or
                    (yesterday_feature == 5 and times >= 8) or
                    (yesterday_feature == 6 and times >= 6) or
                    (10 >= yesterday_feature >= 7 and times >= 5) or
                    (yesterday_feature > 10 and times >= 2.5) or
                    (yesterday_feature > 20 and times >= 2) or
                    (yesterday_feature > 35 and times >= 1.5) or
                    (yesterday_feature > 50 and diff >= 25)):
                is_oestrus = 1
                is_breeding = 2
                confidence = 0.8

        # ---------- 发情 ----------
        if len(features) >= 3:
            before_yesterday_feature = features[-3]

            if today_feature >= yesterday_feature >= before_yesterday_feature and today_feature!=0 and today_feature>=25:
                is_oestrus = 1
                is_breeding = 2
                confidence = max(confidence, 0.8)
                if dizeng_beishu(before_yesterday_feature, yesterday_feature) or dizeng_beishu(yesterday_feature,
                                                                                               today_feature):
                    confidence = 1

        # # ---------- 配种 ----------
        # if len(features) >= 4:
        #     before_yesterday_feature = features[-3]
        #     before_before_yesterday_feature = features[-4]
        #
        #     if today_feature >= yesterday_feature >= before_yesterday_feature >= before_before_yesterday_feature and today_feature != 0:
        #         is_breeding = 2
        #         confidence = max(confidence, 0.8)
        #
        #         if dizeng_beishu(before_before_yesterday_feature, before_yesterday_feature) or dizeng_beishu(
        #                 before_yesterday_feature, yesterday_feature) or dizeng_beishu(yesterday_feature,
        #                                                                               today_feature):
        #             confidence = 1

        return is_oestrus, is_breeding, confidence


def dizeng_beishu(yesterday_feature, today_feature):
    if yesterday_feature == -1 or today_feature == -1:
        return False

    else:
        if yesterday_feature==0:
            if today_feature>=40:
                return True

        else:
            # 如果今天的活动量是昨天的两倍，则配种
            times = today_feature / yesterday_feature
            if (yesterday_feature<5 and times>20) or (yesterday_feature>=5 and times>=2):
                return True

    return False

