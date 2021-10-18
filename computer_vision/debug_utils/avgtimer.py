import time

class AvgTimer:
    """
    Simple timer to get average elapsed time.

    Usage:

    avgtimer = AvgTimer(rolling_length = 100)

    for frame in video:
        avgtimer.start("preprocess")
        preprocess()
        avgtimer.end("preprocess")
        .
        .
        .
        print("Rolling average of preprocessing time: {avgtimer.rolling_avg("preprocess")}")
    """
    def __init__(self, rolling_length = 100):
        self.in_progress = {}
        self.elapsed_history = {}
        self.rolling_length = rolling_length
    def start(self, task_name):
        self.in_progress.update({task_name: time.time()})
    def end(self, task_name):
        elapsed_time = time.time() - self.in_progress[task_name] 

        if task_name in self.elapsed_history.keys():
            prev_times = self.elapsed_history[task_name]
            if len(prev_times) >= self.rolling_length:
                prev_times = prev_times[1:]
            prev_times.append(elapsed_time)
            self.elapsed_history.update({task_name: prev_times})
        else:
            self.elapsed_history.update({task_name: [elapsed_time]})
    def rolling_avg(self, task_name):
        try:
            rolling_times = self.elapsed_history[task_name]
        except KeyError:
            rolling_times = [-1]

        rolling_sum = sum(rolling_times)
        length = len(rolling_times)
        rolling_avg = rolling_sum / length
        return rolling_avg
