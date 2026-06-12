import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}
DEMO1 = {
    "annotation_path": "/path/to/ChatSR/qwen-vl-finetune/demo/single_images.json",
    "data_path": "/path/to/ChatSR/qwen-vl-finetune/",
}

# Symbolic regression dataset config
SYMBOLIC_REGRESSION = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_1k = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_2/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_10k = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_10000/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_20W = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_20W/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_1M = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_1M/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_1000_NO_TEXT = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_1000_no_text/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_2w = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_2w/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_2M = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_lexical_20/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_100 = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_symbol_100/val.json",
    "data_path": "",
}

SYMBOLIC_REGRESSION_LEXICAL_2M = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_lexical_2M/train.json",
    "data_path": "",
}

SYMBOLIC_REGRESSION_LEXICAL_POINT_20 = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_20/train.json",
    "data_path": "",
}
SYMBOLIC_REGRESSION_LEXICAL_POINT_20_TEST = {
    "annotation_path": "/path/to/ChatSR/symbolic_regression_data_20/test.json",
    "data_path": "",
}








data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "DEMO1": DEMO1,
    "SYMBOLIC_REGRESSION": SYMBOLIC_REGRESSION,
    "SYMBOLIC_REGRESSION_1k": SYMBOLIC_REGRESSION_1k,
    "SYMBOLIC_REGRESSION_10k": SYMBOLIC_REGRESSION_10k,
    "SYMBOLIC_REGRESSION_20W": SYMBOLIC_REGRESSION_20W,
    "SYMBOLIC_REGRESSION_1M": SYMBOLIC_REGRESSION_1M,
    "SYMBOLIC_REGRESSION_1000_NO_TEXT": SYMBOLIC_REGRESSION_1000_NO_TEXT,
    "SYMBOLIC_REGRESSION_2w": SYMBOLIC_REGRESSION_2w,
    "SYMBOLIC_REGRESSION_2M": SYMBOLIC_REGRESSION_2M,
    "SYMBOLIC_REGRESSION_100": SYMBOLIC_REGRESSION_100,
    "SYMBOLIC_REGRESSION_LEXICAL_2M": SYMBOLIC_REGRESSION_LEXICAL_2M,
    "SYMBOLIC_REGRESSION_LEXICAL_POINT_20": SYMBOLIC_REGRESSION_LEXICAL_POINT_20,
    "SYMBOLIC_REGRESSION_LEXICAL_POINT_20_TEST": SYMBOLIC_REGRESSION_LEXICAL_POINT_20_TEST,
    
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["DEMO1%100"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
