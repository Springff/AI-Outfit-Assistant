## 服装搭配大模型项目下载
```
git clone https://www.modelscope.cn/studios/zoriswang/Clothes_llm_model.git
```


## 阿里云 ecs.g8i.6xlarge 实例运行方法
```
# 下载 oh-my-zsh（非必要）
sudo apt install zsh
chsh -s /bin/zsh

sh -c "$(curl -fsSL https://gitee.com/shmhlsy/oh-my-zsh-install.sh/raw/master/install.sh)"
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting 

## Then change zsh plugin to `zsh-autosuggestions` `zsh-syntax-highlighting`
source ~/.zshrc


# 下载 anaconda
apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash anaconda.sh

# 创建运行的 ipex 环境
conda create -n ipex python=3.10
conda activate ipex
pip install -r requirements.txt

# 下载大模型
python install.py


# 运行服装搭配大模型应用
# 运行前请在 app.py 中输入您的智谱平台的 api_key (40行)

python -m streamlit run app.py 
```

## 项目介绍
1. `./doc` 和 `./data` 为知识库数据，`./img` 为前端的静态照片库
2. `./install.py` 为模型下载代码
3. `./app.py` 为服装搭配大模型的应用运行代码