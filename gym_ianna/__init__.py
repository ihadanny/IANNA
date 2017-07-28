import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='IANNA-v0',
    entry_point='gym_ianna.envs:IANNAEnv',
#    timestep_limit=1000,
#    reward_threshold=1.0,
#    nondeterministic = True,
)