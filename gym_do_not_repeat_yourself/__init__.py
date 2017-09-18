import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DoNotRepeatYourself-v0',
    entry_point='gym_do_not_repeat_yourself.envs:DoNotRepeatYourselfEnv',
#    timestep_limit=1000,
#    reward_threshold=1.0,
#    nondeterministic = True,
)
