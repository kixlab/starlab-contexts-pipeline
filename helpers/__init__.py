PATH = "step_data/"
TASK_ID  = "custom"
META_TITLE = "$meta$"

TASK_DESCRIPTIONS = {
    "custom2": "How to make Pasta Carbonara?",
    "custom3": "How to make Pasta Carbonara?",
    "custom": "How to bake a muffin?",
    "11967": "How to make a paper airplane?",
}

LIBRARY = {
    "custom3": [
        "https://www.youtube.com/watch?v=75p4UHRIMcU",
        "https://www.youtube.com/watch?v=dzyXBU3dIys",
        "https://www.youtube.com/watch?v=D_2DBLAt57c",
        "https://www.youtube.com/watch?v=3AAdKl1UYZs",
        "https://www.youtube.com/watch?v=qoHnwOHLiMk",
        "https://www.youtube.com/watch?v=NqFi90p38N8",
    ],
    "custom2": [
        "https://www.youtube.com/watch?v=75p4UHRIMcU",
        "https://www.youtube.com/watch?v=dzyXBU3dIys",
        "https://www.youtube.com/watch?v=D_2DBLAt57c",
        # "https://www.youtube.com/watch?v=3AAdKl1UYZs",
        # "https://www.youtube.com/watch?v=qoHnwOHLiMk",
        # "https://www.youtube.com/watch?v=NqFi90p38N8",
    ],
    "11967": [
        "https://www.youtube.com/watch?v=yJQShkjNn08",
        "https://www.youtube.com/watch?v=yweUoYP1v_o",
        "https://www.youtube.com/watch?v=Ehntsffsx08",
        "https://www.youtube.com/watch?v=tdk9_Xs_CC0",
        "https://www.youtube.com/watch?v=dkhy4vn9HcY",
        "https://www.youtube.com/watch?v=QECo58lV-bE",
        "https://www.youtube.com/watch?v=SMh2sjuEwxM",
        "https://www.youtube.com/watch?v=DaEzhwLFPi8",
        "https://www.youtube.com/watch?v=J_5scvrv0LU",
        "https://www.youtube.com/watch?v=umbBEHlpTfo",
        "https://www.youtube.com/watch?v=pq_INi_4IBI",
        "https://www.youtube.com/watch?v=pYOQutHfCDo",
    ],
    "custom": [
        # "https://www.youtube.com/shorts/B-XGIGS4Ipw", # short
        # "https://www.youtube.com/shorts/fWp5z_YM07Q", # short
        "https://www.youtube.com/watch?v=aEFvNsBDCWs", # has verbal
        "https://www.youtube.com/watch?v=gN-orgrgvU8", # has verbal
        "https://www.youtube.com/watch?v=cZ2KJPGVwNU", # has verbal
    ]
}

ALIGNMENT_DEFINITIONS = {
    "`Additional Information`": "what is new",
    "`Alternative Method`": "how is the method different",
    "`Alternative Setting`": "how is the setting different",
    "`Alternative Example`": "how are both method and setting different",
}

VIDEO_SETS = {
    "seen": "previously seen",
    "unseen": "user have not seen",
}

def str_to_float(str_time):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], str_time.split(":")))

def float_to_str(float_time):
    return str(int(float_time // 3600)) + ":" + str(int((float_time % 3600) // 60)) + ":" + str(int(float_time % 60))