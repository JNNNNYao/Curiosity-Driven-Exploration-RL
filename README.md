# Curiosity-Driven-Exploration-RL
reimplement practice of noreward-rl
+ [Paper: Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)
+ [Original code: noreward-rl](https://github.com/pathak22/noreward-rl)
## Dependencies
```bash
apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev libopenal-dev timidity libwildmidi-dev unzip

apt-get install libboost-all-dev

apt-get install liblua5.1-dev

pip install vizdoom
```
## Usage
+ training
```bash
# after install all the dependencies
cd Curiosity-Driven-Exploration-RL/src/
python3 train.py -s dense
```
+ demo
```bash
# after install all the dependencies
cd Curiosity-Driven-Exploration-RL/src/
python3 demo.py -s dense -p ../ckpt/dense/ICM.ckpt -g -f
```
## Result
+ sparse
<img src="gif/sparse.gif" width="300">

+ verySparse
<img src="gif/verySparse.gif" width="300">
