from pettingzoo.atari import pong_v2

# env = pong_v2.parallel_env(num_players=2)
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class='stable_baselines3')
# model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
# model.learn(total_timesteps=20000)
# model.save('policy')


# frame_list = []
        
env = pong_v2.env()
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)
# model = PPO.load('policy')
# env.reset()
# for agent in env.agent_iter(max_iter=3*2**10):
#    obs, reward, done, info = env.last()
#    act = model.predict(obs, deterministic=True)[0] if not done else None
#    env.step(act)
#    frame_list.append(PIL.Image.fromarray(env.render(mode='rgb_array')))
# frame_list[0].save('out.gif', save_all=True, append_images=frame_list[1:], duration=3, loop=0)
