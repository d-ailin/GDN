import time
import math
from datetime import datetime
from pytz import utc, timezone

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSincePlus(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timestamp2str(sec, fmt, tz):
    return datetime.fromtimestamp(sec).astimezone(tz).strftime(fmt)
