# Composing Reinforcement Learning Policies, with Formal Guarantees
Code for replicating the experiments from the paper [*Composing Reinforcement Learning Policies, with Formal Guarantees*](https://arxiv.org/abs/2402.13785).

See also our [blogpost](https://delgrange.me/post/composing_rl/).

![Demo](assets/fusing_drl_components.gif)

## Dependencies
You may find the pip dependencies in the file `requirements.txt`.
Code tested on `python 3.9.6`. 

## Environments
The [Grid world](https://youtu.be/crowN8-GaRg) and [ViZDoom](https://delgrange.me/post/composing_rl/video.mp4) environments described in the paper are available in `reinforcement_learning/environments`.

## Replicating the paper results
### Training WAE-DQN policies
```
cd reinforcement_learning
./train_directions.sh
```
### Train baseline DQN policies
```
cd reinforcement_learning
./train_baselines.sh
```
### Synthesis from WAE-DQN policies
```
./synthesis/synth_doom.sh
./synthesis/synth_pacman.sh
```
### Compute PAC bounds
```
./synthesis/pac_bounds.sh
```
### Pre-trained models
Pre-trained low-level policies can be found in the folder `reinforcement_learning/saves`.

## Cite
If you use this code, please cite it as:
```
@inproceedings{
  DALSNP2025composing,
  title={Composing Reinforcement Learning Policies, with Formal Guarantees},
  author={Florent Delgrange and Guy Avni and Anna Lukina and Christian Schilling and Ann Nowe and Guillermo Perez},
  booktitle={Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems, Detroit, Michigan, USA, May 19-23, IFAAMAS},
  year={2025},
}
```
