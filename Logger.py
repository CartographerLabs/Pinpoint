from datetime import datetime


class logger():
    """
    A  wrapper class around the Python print function used to only print
    """
    DEBUG = False

    @staticmethod
    def print_message(message, logging_level = 0):
        """
        A  wrapper function around the Python print function used to only print
        :param message: the message to print
        :param override_debug: a boolean on if the DEBUG status should be override. if True a log will be printed,
        irrespective of if in Debug mode.
        """
        if logging_level > 0 or logger.DEBUG:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("{} | {}".format(current_time,message))