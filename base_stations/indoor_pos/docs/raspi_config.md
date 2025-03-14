## Headless Setup
1. Install Raspberry Pi OS on microSD card, enable ssh

2. Remove known host from your ssh list
    ```sh
    ssh-keygen -R raspi-02.local
    ```
3. Verify deletion
    ```sh
    cat ~/.ssh/known_hosts
    ```
4. Connect to raspberry pi (e.g. hostname: raspi-01, username: admin, password: 12341234)
    ```sh
    ssh admin@raspi-01.local
    ```
5. Enable VNC server
    ```sh
    sudo raspi-config
    # Go to -> 3 Interface Options -> I2 VNC
    ```
6. Generate [ssh key for GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux), or [personal access tokens](https://github.com/settings/tokens)
    ```sh
    # Generate ssh key on raspberry pi and then add it to github
    ssh-keygen -t ed25519 -C "your_email@example.com"
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519
    cat ~/.ssh/id_ed25519.pub
   
   # Generate personal access token on github and then add it to raspberry pi
   echo "your_personal_access_token" > ~/.github_pat
   chmod 600 ~/.github_pat 
   echo 'export GITHUB_PAT=$(cat ~/.github_pat)' >> ~/.bashrc
   source ~/.bashrc
   echo $GITHUB_PAT
   ```
7. Go to [GitHub Key Settings](https://github.com/settings/keys
) and add new ssh key

8. Clone the repository from GitHub
   ```sh
   git clone git@github.com:ucph-ccs/mbox-video.git
   # or
   git clone https://$GITHUB_PAT@github.com/ucph-ccs/mbox-video.git
   ```
   
9. Install [miniforge](https://github.com/conda-forge/miniforge#download) or  [miniconda](https://docs.anaconda.com/free/miniconda/index.html)
    ```sh
    # For Raspberry Pi
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
       
    # For Mac and Linux
    wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname)-$(uname -m).sh"
    bash Miniconda3-$(uname)-$(uname -m).sh
    ```
