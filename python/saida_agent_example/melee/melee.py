# Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
# Modified for SCV Resource Gathering & Building task with DQN.
# Compatible with Python 3.5.6

import os
import numpy as np
from datetime import datetime
import math
import argparse

# SAIDA_RL 프레임워크의 핵심 컴포넌트들을 import 합니다.
from core.algorithm.DQN import DQNAgent
from core.memories import SequentialMemory
from core.policies import LinearAnnealedPolicy, EpsGreedyQPolicy, GreedyQPolicy
from core.common.processor import Processor
from core.callbacks import DrawTrainMovingAvgPlotCallback
from core.common.util import OPS

# Keras/TensorFlow 관련 라이브러리들을 import 합니다.
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam

# C++로 구현된 실제 게임 환경과 연결될 Python 래퍼를 import 합니다.
from saida_gym.starcraft.melee import melee

# --- 1. 명령행 인자 설정 ---
parser = argparse.ArgumentParser(description='DQN Configuration for SCV Task')
parser.add_argument(OPS.NO_GUI.value, help='Run without GUI', type=bool, default=False)
parser.add_argument(OPS.DOUBLE.value, help='Enable Double DQN', default=True, action='store_true')
parser.add_argument(OPS.DUELING.value, help='Enable Dueling DQN', default=True, action='store_true')
parser.add_argument(OPS.BATCH_SIZE.value, type=int, default=128, help="Batch size")
parser.add_argument(OPS.REPLAY_MEMORY_SIZE.value, type=int, default=50000, help="Replay memory size")
parser.add_argument(OPS.LEARNING_RATE.value, type=float, default=0.0005, help="Learning rate")
parser.add_argument(OPS.TARGET_NETWORK_UPDATE.value, type=int, default=1000, help="Target network update interval")
parser.add_argument(OPS.WINDOW_LENGTH.value, type=int, default=1, help="Window length for sequential observations")
# parser.add_argument(OPS.LOAD_MODEL.value, help='Load pre-trained model', default=False, action='store_true')
# parser.add_argument(OPS.MODEL_NAME.value, help='Name of the model to load', type=str, default="")

args = parser.parse_args()
dict_args = vars(args)

# --- 파일명 생성을 위한 접미사 (Python 3.5 호환) ---
post_fix = ''
for k, v in dict_args.items():
    if k in [OPS.NO_GUI()]:
        continue
    # f-string 대신 .format() 사용
    post_fix += '_{key}_{value}'.format(key=k, value=v)

# --- 2. 하이퍼파라미터 설정 ---
NO_GUI = dict_args[OPS.NO_GUI()]
# LOAD_MODEL = dict_args[OPS.LOAD_MODEL()]
# MODEL_NAME = dict_args[OPS.MODEL_NAME()]

# DQN 관련 하이퍼파라미터
NB_STEPS = 2000000
DISCOUNT_FACTOR = 0.99
ENABLE_DOUBLE = dict_args[OPS.DOUBLE()]
ENABLE_DUELING = dict_args[OPS.DUELING()]
BATCH_SIZE = dict_args[OPS.BATCH_SIZE()]
REPLAY_BUFFER_SIZE = dict_args[OPS.REPLAY_MEMORY_SIZE()]
LEARNING_RATE = dict_args[OPS.LEARNING_RATE()]
TRAIN_INTERVAL = 4
TARGET_MODEL_UPDATE_INTERVAL = dict_args[OPS.TARGET_NETWORK_UPDATE()]
WINDOW_LENGTH = dict_args[OPS.WINDOW_LENGTH()]

STATE_SIZE = 110
ACTION_SIZE = 10

# --- 3. 상태 및 보상 처리를 위한 클래스 정의 ---
class ScvProcessor(Processor):
    def __init__(self, state_size):
        self.state_size = state_size
        self.last_action = -1
        self.episode_count = 0
        self.total_wins = 0
        self.cumulate_reward = 0.0

    def get_config(self):
        return {'state_size': self.state_size}

    def process_step(self, observation, reward, done, info):
        self.cumulate_reward += reward
        state_array = self.process_observation(observation)

        if done:
            self.episode_count += 1
            if reward > 100:
                self.total_wins += 1
            
            win_rate = (self.total_wins / self.episode_count) * 100 if self.episode_count > 0 else 0.0
            # f-string 대신 .format() 사용
            print("Episode {} finished. Cumulative Reward: {:.2f}, Win Rate: {:.2f}%".format(self.episode_count, self.cumulate_reward, win_rate))

            self.cumulate_reward = 0.0

        return state_array, reward, done, info

    def process_observation(self, observation, **kwargs):
        s = np.zeros(self.state_size)
        idx = 0

        def set_unit_features(unit_obs):
            nonlocal idx
            s[idx] = unit_obs.pos_x / 1024.0
            s[idx+1] = unit_obs.pos_y / 1024.0
            s[idx+2] = unit_obs.hp / (unit_obs.max_hp + 1e-6)
            s[idx+3] = 1.0 if unit_obs.is_gathering else 0.0
            s[idx+4] = 1.0 if unit_obs.is_constructing else 0.0
            s[idx+5] = 1.0 if unit_obs.is_idle else 0.0
            s[idx+6] = unit_obs.ground_weapon_cooldown / 15.0
            idx += 10

        if len(observation.my_unit) > 0:
            set_unit_features(observation.my_unit[0])
        else:
            idx += 10

        num_other_units = 1 + 8 + 1
        for i in range(num_other_units):
            if i < len(observation.en_unit):
                set_unit_features(observation.en_unit[i])
            else:
                idx += 10
        return s

# --- 4. DQN 모델 빌더 함수 정의 ---
def build_model(state_size, action_size, dueling):
    model = Sequential()
    model.add(Reshape((state_size * WINDOW_LENGTH,), input_shape=(WINDOW_LENGTH, state_size)))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(action_size, activation='linear', kernel_initializer='he_uniform'))
    
    print("DQN Model Summary:")
    model.summary()
    return model

# --- 5. 메인 실행 로직 ---
if __name__ == '__main__':
    training_mode = True
    # f-string 대신 .format() 사용
    FILE_NAME = "SCV_DQN-{}".format(datetime.now().strftime('%m%d-%H%M'))
    
    env = melee(frames_per_step=8, no_gui=NO_GUI)

    try:
        processor = ScvProcessor(state_size=STATE_SIZE)
        model = build_model(STATE_SIZE, ACTION_SIZE, ENABLE_DUELING)
        memory = SequentialMemory(limit=REPLAY_BUFFER_SIZE, window_length=WINDOW_LENGTH)
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=500000)
        test_policy = GreedyQPolicy()

        agent = DQNAgent(model=model, nb_actions=ACTION_SIZE, memory=memory, processor=processor,
                         policy=policy, test_policy=test_policy, enable_double_dqn=ENABLE_DOUBLE,
                         enable_dueling_network=ENABLE_DUELING, train_interval=TRAIN_INTERVAL,
                         gamma=DISCOUNT_FACTOR, batch_size=BATCH_SIZE,
                         target_model_update=TARGET_MODEL_UPDATE_INTERVAL)
        
        agent.compile(Adam(lr=LEARNING_RATE), metrics=['mae'])

        # if LOAD_MODEL:
        #    # f-string 대신 .format() 사용
        #    print("Loading model weights from: {}".format(MODEL_NAME))
        #    agent.load_weights(MODEL_NAME)

        # f-string 대신 .format() 사용
        graph_path = os.path.realpath('../../save_graph/{}{}.png'.format(FILE_NAME, post_fix))
        cb_plot = DrawTrainMovingAvgPlotCallback(graph_path, 100, 50, l_label=['episode_reward'])
        
        # 학습 시작. SAIDA-RL 버전에 따라 agent.fit 또는 agent.run을 사용합니다.
        # agent.fit(env, nb_steps=NB_STEPS, visualize=False, verbose=2, callbacks=[cb_plot])
        agent.run(env, nb_steps=NB_STEPS, train_mode=training_mode, verbose=2, callbacks=[cb_plot])


        if training_mode:
            save_path = os.path.realpath("../../save_model")
            model_save_name = '{}{}'.format(FILE_NAME, post_fix)
            # f-string 대신 .format() 사용
            print("Saving final model to: {}".format(os.path.join(save_path, model_save_name)))
            agent.save_weights('{}.h5f'.format(os.path.join(save_path, model_save_name)), overwrite=True)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
        if training_mode:
            save_path = os.path.realpath("../../save_model")
            model_save_name = '{}_interrupted'.format(FILE_NAME, post_fix)
            # f-string 대신 .format() 사용
            print("Saving interrupted model to: {}".format(os.path.join(save_path, model_save_name)))
            agent.save_weights('{}.h5f'.format(os.path.join(save_path, model_save_name)), overwrite=True)

    finally:
        env.close()