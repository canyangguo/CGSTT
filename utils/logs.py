import datetime

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

def log_string(log, info, use_info=True, pnt=True):  # p decide print
    if use_info:
        info = '{} - INFO - {}'.format(get_local_time(), info)

    log.write('\n' + info)
    log.flush()
    if pnt:
        print(info)