import configparser
import numpy as np
import ast

# Currently only works for floats, ints and lists
# Beware PATH!
class Hyperparameters:
    def __init__(self, config_file = 'hyperparameters.ini'):
        config = configparser.ConfigParser()
        config.read(config_file)
        if not config.read(config_file):
            raise Exception("Could not read config file")

        for key in config['Hyperparameters']:
            try:
                value = int(config['Hyperparameters'][key])
            except ValueError:
                try:
                    value = float(config['Hyperparameters'][key])
                except ValueError:
                    try:
                        value = np.array(list(config['Hyperparameters'][key]))
                    except ValueError:
                        value = config['Hyperparameters'][key]
            setattr(self, key, value)


def hyperparams_dict(section):
    config = configparser.ConfigParser()
    config.read('hyperparameters.ini')
    if not config.read('hyperparameters.ini'):
        raise Exception("Could not read config file")
    
    params = config[section]
    typed_params = {}
    for key, value in params.items():
        try:
            # Attempt to evaluate the value to infer type
            typed_params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # Fallback to the original string value if evaluation fails
            typed_params[key] = value
    
    return typed_params

