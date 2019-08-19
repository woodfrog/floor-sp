import yaml
from pdb import set_trace as st


def load_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = yaml.load(f)
    return config_dict


class Struct:
    def __init__(self, **entries):
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = Struct(**v)
            elif isinstance(v, list):
                rv = []
                for item in v:
                    if isinstance(item, dict):
                        rv.append(Struct(**item))
                    else:
                        rv.append(item)
            else:
                rv = v
            rec_entries[k] = rv
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, Struct):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "struct {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "Struct(%r)" % self.__dict__


def compose_config_str(configs, keywords, extra=None):
    str_list = list()
    for keyword in keywords:
        if hasattr(configs, keyword):
            str_list.append(keyword + '_' + str(getattr(configs, keyword)))
    configs_str = '_'.join(str_list)
    if extra is not None:
        configs_str += '_' + extra
    return configs_str


if __name__ == '__main__':
    config_dic = load_config('./configs/sample_config.yaml')
    configs = Struct(**config_dic)
    print(config_dic)
