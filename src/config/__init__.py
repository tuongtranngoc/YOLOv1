import yaml

def get_config(yaml_pth='config.yml'):
    try:
        with open(yaml_pth, 'r', encoding='utf-8') as f_cfg:
            config = yaml.safe_load(f_cfg)
    except yaml.YAMLError as e:
        print(e)

    return config

CFG_PATH = 'src/config/config.yml'
CFG = get_config(CFG_PATH)