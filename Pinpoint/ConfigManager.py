import json
from pathlib import Path


class ConfigManager:
    """
    A wrapper file used to abstract Twitter config options.    """

    @staticmethod
    def _get_config(config_path):

        if Path(config_path).is_file() == False:
            raise Exception("The {} config file was not found.".format(config_path))

        with open(config_path) as json_file:
            twitter_config_dict = json.load(json_file)

        return twitter_config_dict

    @staticmethod
    def getTwitterConfig():
        return ConfigManager._get_config("twitterConfig.json")