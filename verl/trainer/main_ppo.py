# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

#################################
import asyncio
import logging
import os
import subprocess
import sys
import time

import hydra
import ray
from ray import WORKER_MODE

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

logger = logging.getLogger(__name__)
for env in ["TRITON_CACHE_DIR", "TRITON_AUTOTUNE_CACHE_DIR", "TRITON_HOME", "XDG_CACHE_HOME"]:
    if env in os.environ:
        os.makedirs(os.environ[env], exist_ok=True)


def is_driver():
    return ray.get_runtime_context().worker.mode != WORKER_MODE if ray.is_initialized() else True


def get_driver_rank():
    assert is_driver(), "this function should only be run on a driver"
    return int(os.getenv("RANK", "0"))


def get_driver_world_size():
    assert is_driver(), "this function should only be run on a driver"
    return int(os.getenv("WORLD_SIZE", "1"))


def get_driver_master_addr():
    assert is_driver(), "this function should only be run on a driver"
    return os.getenv("MASTER_ADDR", "127.0.0.1")


def get_driver_master_port():
    assert is_driver(), "this function should only be run on a driver"
    return os.getenv("MASTER_PORT", "6379")


def get_driver_node_name():
    assert is_driver(), "this function should only be run on a driver"
    return os.getenv("WORKER_ID", f"{get_driver_master_addr()}:{get_driver_rank()}")


def get_driver_dashboard_port():
    assert is_driver(), "this function should only be run on a driver"
    return os.getenv("DASHBOARD_PORT", "8265")


def is_multi_tenant():
    return os.getenv("MULTI_TENANT", "0") == "1"


def execute(cmd, check=False, retry=1):
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
    state = ret.returncode == 0
    msg = ret.stdout if state else ret.stderr
    if not state:
        logger.warning(f"execute {cmd} got error {msg}")
        if retry > 1:
            logger.warning(f"retry {cmd} ...")
            time.sleep(1)
            return execute(cmd, check, retry - 1)
    return state, msg


def is_connection_refused(msg):
    keywords = ["StatusCode.UNAVAILABLE", "Connection refused", "failed to connect to all addresses"]
    return any(keyword in msg for keyword in keywords)


def get_ray_status(retry=3):
    cluster_state, msg = execute("ray status", retry=retry)
    if cluster_state:
        return True, None
    elif is_connection_refused(msg):
        return False, msg
    if not cluster_state:
        return False, msg
    return True, msg


def filter_known_msg(msg):
    if "StatusCode.DEADLINE_EXCEEDED" in msg:
        return True
    return False


def is_ray_cluster_running():
    if is_multi_tenant():
        ret = subprocess.run(
            f"ray status --address {get_driver_master_addr()}:{get_driver_master_port()}",
            shell=True,
            capture_output=True,
        )
    else:
        ret = subprocess.run(f"ray status", shell=True, capture_output=True)
    if ret.returncode != 0:
        return False
    return True


def wait_for_nodes(expected):
    # Wait for all nodes to join the cluster.
    while True:
        nodes_info = ray.nodes()
        active_nodes = [node for node in nodes_info if node["Alive"]]
        num_nodes = len(active_nodes)
        if num_nodes != expected:
            logger.info(f"{num_nodes} nodes have joined so far, waiting for {expected - num_nodes}.")
            time.sleep(1)
        else:
            break


# def start_ray_cluster():
#     rank = get_driver_rank()
#     world_size = get_driver_world_size()
#     master_addr = get_driver_master_addr()
#     master_port = get_driver_master_port()
#     node_name = get_driver_node_name()
#     dashboard_port = get_driver_dashboard_port()

#     # --- [修改点：定义你自己的 Ray 运行目录] ---
#     custom_ray_temp = "/srv/user/zhujs/work/ray_runtime" 
#     os.makedirs(custom_ray_temp, exist_ok=True)

#     if is_ray_cluster_running():
#         logger.info("Ray cluster already initialized")
#         return False

#     if rank == 0:
#         # 添加 --temp-dir 参数并禁用统计
#         cmd = (f"ray start --head --port={master_port} --node-name={node_name} "
#                f"--dashboard-port={dashboard_port} --temp-dir={custom_ray_temp} "
#                f"--disable-usage-stats")
#     else:
#         time.sleep(60)
#         # Worker 节点也要指向同一个 temp-dir (如果是单机则无所谓，分布式必须一致)
#         cmd = (f"ray start --address={master_addr}:{master_port} --node-name={node_name} "
#                f"--dashboard-port={dashboard_port} --temp-dir={custom_ray_temp}")

#     logger.info(f"Starting ray cluster with custom temp dir: {cmd}")
#     ret = subprocess.run(cmd, shell=True, capture_output=True)
#     if ret.returncode != 0:
#         logger.error(f"Failed to start ray cluster: {cmd}")
#         logger.error(f"ret.stdout: {ret.stdout}")
#         logger.error(f"ret.stderr: {ret.stderr}")
#         sys.exit(1)
#     return True

def start_ray_cluster():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port() 
    node_name = get_driver_node_name()
    dashboard_port = get_driver_dashboard_port()

    # 依然建议使用自定义目录，避免与系统 /tmp/ray 冲突
    custom_ray_temp = "/srv/user/zhujs/work/ray_runtime" 
    os.makedirs(custom_ray_temp, exist_ok=True)

    if is_ray_cluster_running():
        logger.info("Ray cluster already initialized on 6379")
        return True, master_port, custom_ray_temp

    if rank == 0:
        # 显式添加 --disable-usage-stats 避免启动卡住
        cmd = (f"ray start --head --port={master_port} --node-name={node_name} "
               f"--dashboard-port={dashboard_port} --temp-dir={custom_ray_temp} "
               f"--disable-usage-stats")
    else:
        time.sleep(5) # 稍微等待 Head 启动
        cmd = (f"ray start --address={master_addr}:{master_port} --node-name={node_name} "
               f"--dashboard-port={dashboard_port} --temp-dir={custom_ray_temp}")

    logger.info(f"Starting ray cluster: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Failed to start ray cluster: {ret.stderr.decode()}")
        sys.exit(1)
        
    return True, master_port, custom_ray_temp

def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


# def run_ppo(config) -> None:
#     # TODO(linjunrong.ocss884): this ENV is left for resolving SGLang conflict with ray devices
#     # isolation, will solve in the future
#     os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
#     if not ray.is_initialized():
#         # this is for local ray cluster
#         ray.init(
#             runtime_env={
#                 "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
#             },
#             num_cpus=config.ray_init.num_cpus,
#         )

#     runner = TaskRunner.remote()
#     ray.get(runner.run.remote(config))

def run_ppo(config) -> None:
    # def init():
    RAY_NAMESPACE = "AgentZoom" 
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()

    manual_start = start_ray_cluster()

    runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}}

    if not ray.is_initialized():
        ray.init(
            address=f"{master_addr}:{master_port}" if manual_start else None,
            namespace=RAY_NAMESPACE,
            ignore_reinit_error=True,
            log_to_driver=True,
            runtime_env=runtime_env,
            num_cpus=config.ray_init.num_cpus,
        )
        logger.info("Ray cluster initialized")

    # if rank == 0:
    #     if manual_start:
    #         wait_for_nodes(expected=world_size)
    #     logger.info(f"All nodes joined. Resources: {ray.available_resources()}")
    #     runner = TaskRunner.remote()
    #     ray.get(runner.run.remote(config))
    #     logger.info("Training finished on driver (rank0).")
    #     return
    if rank == 0:
        if manual_start:
            wait_for_nodes(expected=world_size)
        # from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        # head_node_id = ray.get_runtime_context().get_node_id()
        # runner = TaskRunner.options(
        #     scheduling_strategy=NodeAffinitySchedulingStrategy(head_node_id, soft=False),
        #     # name="TaskRunner",
        #     # lifetime="detached",
        # ).remote()
        
        # --- [新增：自动清理同名 Actor 逻辑] ---
        try:
            # 尝试寻找是否已经有叫 TaskRunner 的 Actor 在运行
            existing_runner = ray.get_actor("TaskRunner")
            logger.info("Found existing TaskRunner, killing it to start fresh...")
            ray.kill(existing_runner)
        except ValueError:
            # 如果没找到，说明是干净的环境，直接继续
            pass
        
        # 实例化新的 TaskRunner
        logger.info("Creating new TaskRunner...")

        runner = TaskRunner.options(name="TaskRunner", lifetime="detached").remote()
        try:
            ray.get(runner.run.remote(config))
            print("available:", ray.available_resources())
            print("cluster:", ray.cluster_resources())

        except Exception:
            logger.exception("TaskRunner failed")
            # 给平台一点时间把日志刷出来，避免立刻回收导致 ray status 失败
            time.sleep(10)
            raise

        # 训练正常结束也别立刻死（可选）
        time.sleep(10)
        return
    # rank>0 不提交任务、不退出：保持进程存活，让 ray worker node 持续在线
    logger.info(f"Rank {rank} running as Ray worker node; waiting...")
    while True:
        time.sleep(60)

    # if manual_start:
    #     wait_for_nodes(expected=world_size)

    # logger.info(f"Current ray cluster resources: {ray.available_resources()}")
    # runner = TaskRunner.remote()
    # ray.get(runner.run.remote(config))

    # # if manual_start and rank > 0:
    # #     sys.exit(0)


# @ray.remote(num_cpus=1, num_gpus=0.005)  # please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1, num_gpus=0)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == "naive":
            from verl.workers.reward_manager import NaiveRewardManager

            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == "naive_api":
            from verl.workers.reward_manager import NaiveLLMRewardManager

            reward_manager_cls = NaiveLLMRewardManager
        elif reward_manager_name == "prime":
            from verl.workers.reward_manager import PrimeRewardManager

            reward_manager_cls = PrimeRewardManager
        elif reward_manager_name == "batch":
            from verl.workers.reward_manager import BatchRewardManager

            reward_manager_cls = BatchRewardManager
        elif reward_manager_name == "dapo":
            from verl.workers.reward_manager import DAPORewardManager

            reward_manager_cls = DAPORewardManager
        else:
            raise NotImplementedError

        compute_score = get_custom_reward_fn(config)
        reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer, num_examine=1, compute_score=compute_score, reward_fn_key=config.data.reward_fn_key
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        logger.info("before create trainer")
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        print("after create trainer, before init_workers")
        trainer.init_workers()
        print("after init_workers, before fit")
        trainer.fit()
        print("after fit")
        # trainer.init_workers()
        # trainer.fit()


if __name__ == "__main__":
    # import sys

    # print("################")
    # print("hello world")
    # print("################")
    # sys.exit(-1)
    import os

    import torch

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())

    main()
