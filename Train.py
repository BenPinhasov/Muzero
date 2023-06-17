import gymnasium as gym

from monte_carlo_tree_search import *
from game import *
from replay_buffer import *
from self_play import *
from muzero_model import *
from self_play import report, generate_config_file

# # # unquote to print complet tensor
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

# # # set game environment from gym library
# # # render_mode should be set to None if you don't want rgb observation
env = gym.make("CartPole-v1", render_mode=None)
# # # if you want want rgb observation set render_mode to "rgb_array" , "human" or the render_mode rgb of our env
# # # so for example if you want to use vision_model, you will have to change it
# env = gym.make("ALE/Asterix-v5",render_mode='human')

# # # the random seed are set to 0 for reproducibility purpose
# # #  https://pytorch.org/docs/stable/notes/randomness.html
seed = 0
np.random.seed(seed)  # set the random seed of numpy
torch.manual_seed(seed)  # set the random seed of pytorch

# # # init/set muzero model for training and inference
muzero = Muzero(model_structure='mlp_model',
                # 'vision_model' : will use rgb as observation , 'mlp_model' : will use game state as observation
                observation_space_dimensions=env.observation_space,  # dimension of the observation
                action_space_dimensions=env.action_space,  # dimension of the action allow (gym box/discrete)
                state_space_dimensions=31,
                # support size / encoding space (keep state smaller than hidden layer and use odd number)
                hidden_layer_dimensions=64,  # number of weight in the recursive layer of the mlp
                number_of_hidden_layer=4,  # number of recusion layer of hidden layer of the mlp
                k_hypothetical_steps=10,
                # number of future step you want to be simulate during train (they are mainly support loss)
                learning_rate=1e-2,  # learning rate of the optimizer
                optimizer="adam",  # optimizer "adam" or "sgd"
                lr_scheduler="cosineannealinglr",  # learning rate scheduler
                loss_type="general",  # muzero loss can be "general" or "game"
                num_of_epoch=1000,  # number of step use by lr_scheduler
                device="cpu",
                # device on which you want the compute to be made : "cpu" , "cuda" (it will auto scale on multi gpu or cpu for training and inference)
                type_format=torch.float32,
                # choice the dtype of the model. look at [https://pytorch.org/docs/1.8.1/amp.html#ops-that-can-autocast-to-float16]
                load=False,  # function to load a save model
                use_amp=False,
                # use mix precision (will get more accuracy than single single precision for smaller dtype like torch.float16. amp do not support torch.float64. will turn amp to True by fault for torch.float16)
                bin_method="uniform_bin",
                # "linear_bin" , "uniform_bin" : will have a regular incrementation of action or uniform sampling(pick randomly) from the bound
                bin_decomposition_number=10)  # number of action to sample from low/high bound of a gym discret box

# # # init/set the game storage(stor each game) and dataset(create dataset) generate during training
replay_buffer = ReplayBuffer(window_size=500,  # number of game store in the buffer
                             batch_size=128,  # batch size is the number of observe game during train
                             num_unroll=muzero.k_hypothetical_steps,
                             # number of mouve/play store inside the batched game
                             td_steps=5,  # number of step the value is select and scale on
                             game_sampling="priority",
                             # 'uniform' or "priority" (will game randomly or with a priority distribution)
                             position_sampling="priority")  # 'uniform' or "priority" (will sample position in game randomly or with a priority distribution)

# # # init/set the monte carlos tree search parameter
mcts = Monte_carlo_tree_search(pb_c_base=19652,
                               pb_c_init=1.25,
                               discount=0.997,
                               root_dirichlet_alpha=0.25,
                               root_exploration_fraction=0.25)

# # # ini/set the Game class which embbed the gym game class function
gameplay = Game(gym_env=env,
                discount=mcts.discount,  # should be the same discount than mcts
                limit_of_game_play=500,  # maximum number of mouve , by default float("inf")
                observation_dimension=muzero.observation_dimension,
                action_dimension=muzero.action_dimension,
                rgb_observation=muzero.is_RGB,
                action_map=muzero.action_dictionnary)

print(f"Dimension of the observation space : {muzero.observation_dimension} \
         Dimension of the action space : {muzero.action_dimension}")

# # # train model (if you choice vison model it will render the game by opening and closing window)
epoch_pr, loss, reward, learning_config = learning_cycle(number_of_iteration=1000,
                                                         # number of epoch(step) in  muzero should be the |total amount of number_of_iteration x number_of_training_before_self_play|
                                                         number_of_self_play_before_training=10,
                                                         # number of game played record in the replay buffer before training
                                                         number_of_training_before_self_play=1,
                                                         # number of epoch cpmpute by the model before selplay
                                                         number_of_mcts_simulation=0,
                                                         # number of mcts simulation ( node expension )
                                                         model_tag_number=111,  # tag number use to generate checkpoint
                                                         tempererature_type="static_temperature",
                                                         # "static_temperature" ,"linear_decrease_temperature" ,  "extreme_temperature" and "reversal_tanh_temperature"
                                                         verbose=True,
                                                         # if you want to print the epoch|reward|loss during train
                                                         number_of_worker_selfplay=0,
                                                         # "max" will set the max amount of cpu core, 0 will make selflay run sequentially. Parallelize self-play on the number of worker
                                                         muzero_model=muzero,
                                                         gameplay=gameplay,
                                                         monte_carlo_tree_search=mcts,
                                                         replay_buffer=replay_buffer)

report(muzero, replay_buffer, epoch_pr, loss, reward, verbose=True)

generate_config_file(env, seed, muzero, replay_buffer, mcts, gameplay, learning_config)
