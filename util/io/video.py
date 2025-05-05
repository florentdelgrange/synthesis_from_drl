import os
from typing import Optional, Callable, Union, Collection, Dict, Tuple

import PIL.Image
import imageio
import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing.types import Bool

from PIL import Image, ImageFont, ImageDraw


class VideoEmbeddingObserverNumpy:
    def __init__(self, py_env: PyEnvironment, target_size: Tuple[int, int] = (64, 64)):
        self.py_env = py_env
        self.width, self.height = target_size
        self._data = []

    def __call__(self, *args, **kwargs):
        self._data.append(self.py_env.render(mode='rgb_array'))

    @property
    def data(self):
        return np.stack(self._data, axis=0)


class VideoEmbeddingObserver:
    def __init__(
            self,
            py_env: PyEnvironment,
            file_name: str,
            fps: int = 30,
            num_episodes: int = 1,
            labeling_function: Optional[Callable[[TimeStep], Union[Collection, Dict[str, Collection]]]] = None,
            font_color: str = 'black',
            save_best_only: bool = False,
            output_video_size: Optional[Tuple[int, int]] = None
    ):
        self.py_env = py_env
        self._file_name = file_name
        self.writer = None
        self.fps = fps
        self.best_rewards = -1. * np.inf
        self.cumulative_rewards = 0.
        self.num_episodes = num_episodes
        self.current_episode = 1
        if len(file_name.split(os.path.sep)) > 1 \
                and not os.path.exists(os.path.sep.join(file_name.split(os.path.sep)[:-1])):
            os.makedirs(os.path.sep.join(file_name.split(os.path.sep)[:-1]))
        self.file_name = None
        self.labeling_fn = labeling_function
        self.font_color = font_color
        self.save_best_only = save_best_only
        self.output_video_size = output_video_size

    def finalize(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def resize_image(self, img):
        if self.output_video_size is not None:
            input_height, input_width = img.shape[0:2]
            if isinstance(self.output_video_size, int):
                width = min(self.output_video_size, input_width)
                height = input_height // (input_width // width)
                # to preserve the proportions
            else:
                width, height = self.output_video_size
                width = min(input_width, width)
                height = min(input_height, height)
            return PIL.Image.fromarray(img).resize((width, height)).__array__()
        else:
            return img

    def __call__(self, time_step: TimeStep, *args, **kwargs):
        if self.writer is None:
            self.writer = imageio.get_writer('{}.mp4'.format(self._file_name), fps=self.fps)
        data = self.py_env.render(mode='rgb_array')
        data = self.resize_image(data)

        if self.labeling_fn is not None:
            label = self.labeling_fn(time_step)
            if type(label) is dict:
                label = '\n'.join([str(key)+': '+str(value) for key, value in label.items()])
            else:
                label = str(label)
            img = Image.fromarray(data)
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype('Arial', 10)
            except Exception:
                font = ImageFont.load_default()
            if self.font_color == 'black':
                _fill = (0, 0, 0)
            else:
                _fill = (255, 255, 255)
            draw.text((0, 0), label, font=font, fill=_fill)
            data = np.array(img)

        if data is not None:
            self.writer.append_data(data)
        self.cumulative_rewards += time_step.reward

        if time_step.is_last() and self.current_episode < self.num_episodes:
            self.current_episode += 1
        elif time_step.is_last():
            self.finalize()
            avg_rewards = np.sum(self.cumulative_rewards / self.num_episodes)
            if avg_rewards >= self.best_rewards and self.save_best_only:
                self.best_rewards = avg_rewards
                os.rename('{}.mp4'.format(self._file_name),
                          '{}_rewards={:.2f}.mp4'.format(self._file_name, self.best_rewards))
                self.file_name = '{}_rewards={:.2f}.mp4'.format(self._file_name, self.best_rewards)
            elif self.save_best_only:
                os.remove('{}.mp4'.format(self._file_name))
            else:
                os.rename('{}.mp4'.format(self._file_name),
                          '{}_rewards={:.2f}.mp4'.format(self._file_name, avg_rewards))
                self.file_name = '{}_rewards={:.2f}.mp4'.format(self._file_name, avg_rewards)
            self.cumulative_rewards = 0.
            self.current_episode = 1
