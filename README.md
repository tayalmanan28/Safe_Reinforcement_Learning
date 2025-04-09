# Safe Reinforcement Learning

***

## Overview
Repository containing an implementation of safe reinforcement learning in two custom environments.

While exploring, an RL agent can take actions that lead the system to unsafe states. Here, we use a differentiable RCBF safety layer that minimially alters (in the least-squares sense) the actions taken by the RL agent to ensure the safety of the agent.

## Usage

Following are the list of commands to compile \& run the codes for the various implementations mentioned above:

To install Anaconda follow the instructions in the following webpage:  
https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-20-04-quickstart

Create a conda environment for the Safe RL:  
```
conda create --name safe_rl  
```
Switch to the newly create environment:  
```
conda activate safe_rl  
```
Then, clone the repository on your system:
```
git clone https://github.com/tayalmanan28/Safe_Reinforcement_Learning.git
```

Install the following required packages:
```
pip install -r requirements.txt
```

## Running the Experiments
The environment used in this experiment is `Unicycle`. `Unicycle` involves a unicycle robot tasked with reaching a desired location while avoiding obstacles

### Training: 

* Training the proposed approach: 
```
python3 main.py --gamma_b 20 --max_episodes 200 --cuda --updates_per_step 2 --batch_size 512 --model_based
```

<!-- * Training the baseline:
`python3 main.py --gamma_b 20 --max_episodes 200 --cuda --updates_per_step 1 --batch_size 256 --no_diff_qp`

* Training the modified approach from "End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks": 
`python3 main.py --gamma_b 20 --max_episodes 200 --cuda --updates_per_step 1 --batch_size 256 --no_diff_qp --use_comp True`
 -->
 
### Testing

* To test: 
```
python3 main.py --mode test --resume output/Unicycle-run{1}
```
where `{1}` is the experiment number.
* To Visualize 
```
python3 main.py --mode test --resume output/Unicycle-run{1} --visualize
```

## LICENSE

The code is licenced under the MIT license and free to use by anyone without any restrictions.

***

<p align='center'>Created with :heart: by <a href="https://github.com/tayalmanan28">Manan Tayal</a> </p>
