import subprocess
import queue
import threading
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Need a event loop, that take items from the queue and assign them to workers


def create_file(path):
    dir = os.path.dirname(path)
    if dir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    with open(path, 'w') as f:
        f.write("")


class WorkStarter:
    def make_job(self, param, worker):
        raise NotImplementedError


class CmdWorkStarter(WorkStarter):
    def __init__(self, worker_start_script, logger, make_command):
        self.worker_start_script = worker_start_script
        self.logger = logger
        self.make_command = make_command

    def make_job(self, param, worker):
        def fn():
            cmd = self.make_command(param, worker)
            logger.info(f"Command: {cmd}")
            f = subprocess.Popen(cmd, shell=True)
            f.wait()
        return fn


class WorkerTracker:
    def __init__(self,
                 finished_tasks_dir="finished_tasks",
                 current_tasks=None,
                 finished_tasks=None) -> None:
        self.current_tasks = current_tasks or {}
        self.finished_tasks = finished_tasks or {}
        create_file(finished_tasks_dir,)
        self.finished_task_dir = finished_tasks_dir

    def get_all_finished_tasks_list(self):
        return list(self.finished_tasks.values())

    def start_work(self, worker_address, params):
        self.current_tasks[worker_address] = params

    def finish_work(self, worker_address):
        if worker_address not in self.finished_tasks:
            self.finished_tasks[worker_address] = []
        worker_finished_tasks = self.finished_tasks[worker_address]
        if worker_address in self.current_tasks:
            worker_finished_tasks.append(self.current_tasks[worker_address])
        with open(self.finished_task_dir, 'w') as f:
            json.dump(self.finished_tasks, f)
        del self.current_tasks[worker_address]


class TaskAssigner:
    def __init__(self,
                 param_queue: queue.Queue,
                 ready_workers_queue: queue.Queue,
                 busy_workers_list: list,
                 work_starter: WorkStarter,
                 worker_tracker,
                 logger):
        self.param_queue = param_queue
        self.ready_workers_queue = ready_workers_queue
        self.busy_workers_list = busy_workers_list
        self.work_starter = work_starter
        self.logger = logger
        self.work_tracker = worker_tracker

    def run(self):
        while True:
            param = self.param_queue.get()
            self.logger.info(f"TaskAssigner get task: {param}")
            worker = self.ready_workers_queue.get()
            self.logger.info(f"TaskAssigner get worker: {worker}")
            self.busy_workers_list.append(worker)
            fn_work = self.work_starter.make_job(param, worker)
            threading.Thread(target=fn_work).start()
            self.logger.info(
                f"TaskAssigner assigned task: {param} to worker: {worker}")
            self.work_tracker.start_work(worker, param)


class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self,
                 ready_worker_queue,
                 busy_worker_list: list,
                 logger,
                 worker_tracker,
                 *args, **kwargs):
        self.ready_worker_queue = ready_worker_queue
        self.busy_worker_list = busy_worker_list
        self.worker_tracker = worker_tracker
        self.logger = logger
        super().__init__(*args, **kwargs)

    def do_POST(self):
        self.logger.info(f"Received report from worker: {self.client_address}")
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        worker_info = json.loads(post_data.decode('utf-8'))
        address = worker_info['worker_address']

        self.logger.info(f"Received report: {worker_info}")
        self.worker_tracker.finish_work(address)
        try:
            self.busy_worker_list.remove(address)
            self.ready_worker_queue.put(address)
        except ValueError:
            pass

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Report received")


class Manager:
    def __init__(self, work_starter_factory, finished_task_dir="finished_tasks.json",  is_continue_running=True,) -> None:
        self.logger = logging.getLogger(__name__)
        self.finished_task_dir = finished_task_dir
        self.logger.info(f"Finished task dir: {self.finished_task_dir}")
        self.ready_worker_queue = queue.Queue()
        self.busy_worker_list = []
        self.param_queue = queue.Queue()
        if is_continue_running:
            try:
                with open(self.finished_task_dir, 'r') as f:
                    finished_tasks = json.load(f)
            except Exception as e:
                self.logger.error(e)

                finished_tasks = {}
        else:
            try:
                os.remove(self.finished_task_dir)
            except FileNotFoundError:
                pass
            finished_tasks = {}
        self.worker_tracker = WorkerTracker(finished_tasks_dir=self.finished_task_dir,
                                            finished_tasks=finished_tasks)
        self.work_starter = work_starter_factory(self.logger)
        self.task_assigner = TaskAssigner(self.param_queue,
                                          self.ready_worker_queue,
                                          self.busy_worker_list,
                                          self.work_starter,
                                          self.worker_tracker,
                                          self.logger)
        self.request_handler_factory = lambda *argv: RequestHandler(self.ready_worker_queue,
                                                                    self.busy_worker_list,
                                                                    self.logger,
                                                                    self.worker_tracker, *argv)

    def add_tasks(self, params_list):
        all_finished_tasks = self.worker_tracker.get_all_finished_tasks_list()
        for param in params_list:
            if param not in all_finished_tasks:
                self.param_queue.put(param)

    def add_workers(self, worker_addresses):
        for worker in worker_addresses:
            self.ready_worker_queue.put(worker)

    def run(self):
        threading.Thread(target=self.task_assigner.run).start()
        httpd = HTTPServer(('0.0.0.0', 4443), self.request_handler_factory)
        httpd.serve_forever()


# Usage
if __name__ == "__main__":


    base_params = {
        "learning-rate": 0.1,
        "batch-size": 64,
        "epochs": 100,
        "seed": 0,
        "clip": 0.1,
        "weight-round": "nearest",
        "error-round": "nearest",
        "gradient-round": "nearest",
        "activation-round": "nearest",
        "weight-ew": 2,
        "error-ew": 4,
        "gradient-ew": 4,
        "activation-ew": 2,
        "weight-bw": 3,
        "error-bw": 3,
        "gradient-bw": 3,
        "activation-bw": 3,
        "batchnorm": "id",
        "loss-scale": 1,
        "momentum": 0.7,
        "check-number-ranges": True,
        "mix-precision": True,
        "log-path": "results/loss_scale_batchnorm_clip_experiment",
    }
    params = []
    for rounding in ["nearest", "stochastic"]:
        for loss_scale in [1, 5, 10, 25, 50, 100, 500, 1024, 2048, 4096, 16384, 32768, 65546, 131072]:
            for batchnorm in ["id", "batchnorm"]:
                for clip in [0.01, 0.1, 1]:
                    this_param = base_params.copy()
                    this_param["weight-round"] = rounding
                    this_param["error-round"] = rounding
                    this_param["gradient-round"] = rounding
                    this_param["activation-round"] = rounding
                    this_param["loss-scale"] = loss_scale
                    this_param["batchnorm"] = batchnorm
                    this_param["clip"] = clip
                    params.append(this_param)
    worker_start_script = "bash ~/run_one.sh"

    #workers = [f"gpu{i:02}" for i in range(33, 34)]
    workers = [f"gpu{i:02}" for i in range(1, 19)]
    workers += [f"gpu{i:02}" for i in [26, 27, 28, 29, 30, 31, 33, 34, 35]]

    def make_command(param, worker):
        param_str = ""
        for key, value in param.items():
            param_str += f"--{key} {value} "
        return f"ssh {worker} \"{worker_start_script} {worker} {param_str}\""
    
    def worker_starter_factory(logger): return CmdWorkStarter(
        worker_start_script, logger, make_command=make_command)
    manager = Manager(work_starter_factory=worker_starter_factory)
    manager.add_tasks(params)
    manager.add_workers(workers)
    manager.run()
