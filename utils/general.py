# -*- encoding: utf-8 -*-

from datetime import datetime
from pytz import timezone
import logging

logging.PREDICTVIA_DEBUG_LEVEL_NUM = 21
logging.addLevelName(logging.PREDICTVIA_DEBUG_LEVEL_NUM, "PREDICTVIA")
logging.Logger.predictvia = lambda inst, msg, *args, **kwargs: inst.log(logging.PREDICTVIA_DEBUG_LEVEL_NUM, msg, *args, **kwargs)
logging.predictvia = lambda msg, *args, **kwargs: logging.log(logging.PREDICTVIA_DEBUG_LEVEL_NUM, msg, *args, **kwargs)

logging.basicConfig(level=logging.PREDICTVIA_DEBUG_LEVEL_NUM,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s', )


def TimestampVenezuela():
    """ Return current date as '1970-01-01 23:59:59VET' """
    now_time = datetime.now(timezone("America/Caracas"))
    return now_time.strftime("%Y-%m-%d %H:%M:%SVET")


def log(*args, **kwargs):
    for count, arg in enumerate(args):
        if 'level' in kwargs and kwargs['level'] is not None:
            if kwargs['level'] == 'info':
                logging.info("[%s]: %s" % (TimestampVenezuela(), arg))
            elif kwargs['level'] == 'debug':
                logging.debug("[%s]: %s" % (TimestampVenezuela(), arg))
        else:
            if 'label' in kwargs:
                logging.predictvia("[%s]: \n %s %s" % (TimestampVenezuela(), kwargs['label'], arg))
            else:
                logging.predictvia("[%s]: \n %s" % (TimestampVenezuela(), arg))
