import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # task_suite_name: str = (
    #     "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    # )
    task_suite_name: str = (
        "libero_simple"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_simple
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    # num_trials_per_task: int = 50  # Number of rollouts per task
    num_trials_per_task: int = 10  # Number of rollouts per task (reduced for testing purposes)

    #################################################################################################################
    # Policy parameters
    #################################################################################################################
    # Policy type being served. Used to format observations correctly and name the output directory.
    # Options: "LIBERO", "DROID" 
    policy_type: str = "LIBERO"

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Derive output directory from task suite and policy type
    video_out_path = pathlib.Path(f"data/libero/videos_suite={args.task_suite_name}_pol={args.policy_type}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    video_out_path.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    elif args.task_suite_name == "libero_simple":
        max_steps = 200  # simple 2-item tasks need fewer steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    per_task_results = []  # list of (task_description, successes, episodes)

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states (not needed for libero_simple)
        if args.task_suite_name != "libero_simple":
            initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment and set initial state
            action_plan = collections.deque()
            if args.task_suite_name == "libero_simple":
                # libero_simple has no pre-generated init states; sample fresh each episode
                obs = env.reset()
            else:
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(np.concatenate([img, wrist_img], axis=1))

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        element = _prepare_observation(obs, img, wrist_img, task_description, args.policy_type)

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    if args.policy_type == "DROID":
                        # These policies output joint-space actions (14-dim and 8-dim respectively).
                        # LIBERO env.step() expects 7-dim EEF delta actions, so truncate to 7.
                        action = action[:7]

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                video_out_path / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        per_task_results.append((task_description, task_successes, task_episodes))

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    _write_stats(video_out_path, args, per_task_results, total_successes, total_episodes)


def _prepare_observation(obs, img, wrist_img, task_description, policy_type):
    """Build the observation dict in the format expected by the served policy."""
    if policy_type == "LIBERO":
        state = np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ))
        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": str(task_description),
        }
    elif policy_type == "DROID":
        # DroidInputs expects flat observation keys with joint-space state.
        return {
            "observation/exterior_image_1_left": img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": obs["robot0_joint_pos"],       # (7,)
            "observation/gripper_position": obs["robot0_gripper_qpos"][:1],  # (1,)
            "prompt": str(task_description),
        }


def _write_stats(out_path, args, per_task_results, total_successes, total_episodes):
    """Write evaluation statistics to stats.txt in the output directory."""
    lines = [
        f"Task suite:   {args.task_suite_name}",
        f"Policy type:  {args.policy_type}",
        f"Trials/task:  {args.num_trials_per_task}",
        f"Seed:         {args.seed}",
        "",
        f"Overall success rate: {total_successes}/{total_episodes} "
        f"({total_successes / total_episodes * 100:.1f}%)",
        "",
        "Per-task results:",
    ]
    for task_description, successes, episodes in per_task_results:
        rate = successes / episodes * 100 if episodes > 0 else 0.0
        lines.append(f"  {task_description}: {successes}/{episodes} ({rate:.1f}%)")

    stats_path = out_path / "stats.txt"
    stats_path.write_text("\n".join(lines) + "\n")
    logging.info(f"Stats written to {stats_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
