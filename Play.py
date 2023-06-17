# benchmark 
from self_play import play_game_from_checkpoint, benchmark
import torch

number_of_trial = 100
cache_t, cache_r, cache_a, cache_p = [], [], [], []
for _ in range(number_of_trial):
    tag, reward, action, policy = play_game_from_checkpoint(game_to_play='CartPole-v1',
                                                            model_tag=111,
                                                            model_device="cpu",
                                                            model_type=torch.float32,
                                                            mcts_pb_c_base=19652,
                                                            mcts_pb_c_init=1.25,
                                                            mcts_discount=0.997,
                                                            mcts_root_dirichlet_alpha=0.25,
                                                            mcts_root_exploration_fraction=0.25,
                                                            mcts_with_or_without_dirichlet_noise=True,
                                                            number_of_monte_carlo_tree_search_simulation=11,
                                                            gameplay_discount=0.997,
                                                            temperature=0,
                                                            game_iter=500,
                                                            slow_mo_in_second=0,
                                                            render=True,
                                                            verbose=True,
                                                            benchmark=True,
                                                            render_mode='human')  # Need benchmark True to return output
    # could do it in one list or even wrap the play_game with benchmark but it reduce clarity
    cache_t.append(tag)
    cache_r.append(reward)
    cache_a.append(action)
    cache_p.append(policy)

benchmark(cache_t,
          cache_r,
          cache_a,
          cache_p,
          folder="report",
          verbose=True)

# on cartpole the reward is fix to 1, so it follow the number of mouve.
