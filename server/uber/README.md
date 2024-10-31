# 🌐 OpenMMLA Uber

Uber module of mBox multimodal learning analytic system. For more details, please refer
to [mBox System Design](./docs/mbox_system.md).

## Table of Contents

- [Related Modules](#related-modules)
- [Uber Server Setup](#uber-server-setup)
    - [Clone the repository](#clone-the-repository)
    - [Install required system dependencies](#install-required-system-dependencies)
    - [Install services](#install-services)
        - [Database Server](#database-server)
        - [Message Brokers](#message-brokers)
        - [Load Balancer](#load-balancer)
        - [Streaming Server](#streaming-server)
        - [Dashboard Server](#dashboard-server)
- [Usage](#usage)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

## Related Modules

- [mbox-uber](https://github.com/ucph-ccs/mbox-uber)
- [mbox-audio](https://github.com/ucph-ccs/mbox-audio)
- [mbox-video](https://github.com/ucph-ccs/mbox-video)

## Uber Server Setup

### Clone the repository

   ```bash
   git clone https://github.com/ucph-ccs/mbox-uber.git
   ```

### Install required system dependencies

   ```bash
   # For Mac 
   brew install tmux
   # For Ubuntu
   sudo apt install tmux
   ```

### Install services

The uber server host the following services:
data storage, data messaging, load balancing, data streaming, and web application backend and frontend.

#### Database Server

<details>
<summary>InfluxDB server</summary>

Please refer to [InfluxDB.md](./docs/InfluxDB.md) for detailed setup instructions.

```bash
# Start InfluxDB server
influxd
```

</details>

#### Message Brokers

<details>
<summary>Redis Server</summary>

```bash
brew install redis

# Start Redis server in unprotected mode if running on different machines
redis-server --protected-mode no
```

</details>

<details>
<summary>Mosquitto Server</summary>

```bash
brew install mosquitto

# Edit configuration to listen from all IP addresses
vim /opt/homebrew/etc/mosquitto/mosquitto.conf

# Add/modify these lines
listener 1883 0.0.0.0
allow_anonymous true

# Restart Mosquitto
brew services restart mosquitto
```

</details>

#### Load Balancer

<details>
<summary> Nginx Server </summary>

Please refer to [Nginx Load Balancing](./docs/nginx.md#part-1-load-balancer) for detailed setup
instructions.

</details>

#### Streaming Server

<details>
<summary> RTMP Server </summary>

Please refer to [RTMP configuration](./docs/nginx.md#part-2-rtmp) for detailed setup instructions.

</details>

#### Dashboard Server

<details>
<summary> Backend (Flask) </summary>

1. **Install Conda**
   <details>
   <summary>Conda Installation</summary>

   ```bash
   # For Raspberry Pi
   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   
   # For Mac and Linux
   wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-$(uname)-$(uname -m).sh"
   bash Miniconda3-latest-$(uname)-$(uname -m).sh
   ```
   </details>

2. **Set up Conda environment and install dependencies**
   ```bash
   conda create -n uber-server python=3.10.12 -y
   conda activate uber-server
   pip install -r requirements-dashboard.txt
   ```

</details>   

<details>
<summary> Frontend (React) </summary>

1. **Install Node.js and npm**

   Install [Node.js](https://nodejs.org/en/download/package-manager)

2. **Create Next.js project**
   ```bash
   # npx create-next-app@latest next-react-frontend (if not clone from this repo)
   cd next-react-frontend
   ```

3. **Configure Next.js**

   Edit `next.config.js`, replace `uber-server.local` with your `<uber-server-hostname>` or `<uber-server-IP-adrress>`:
   ```javascript
   module.exports = {
     images: {
       remotePatterns: [
         {
           protocol: 'http',
           hostname: 'uber-server.local',
         },
       ],
     },
     async rewrites() {
       return [
         {
           source: '/api/:path*',
           destination: 'http://localhost:5000/api/:path*',
         },
       ];
     },
   };
   ```

4. **Set environment variables**

   Create `.env.local` file in `/next-react-frontend/`, replace `uber-server.local` with your `<uber-server-hostname>`
   or `<uber-server-IP-address>`:
   ```
   NEXT_PUBLIC_SERVER_IP=uber-server.local
   ```
5. **Install dependencies and start the frontend server**
   ```bash
   rm -rf node_modules package-lock.json
   npm cache clean --force
   npm install
   npm run build
   npm run start
   ```
   
</details>

## Usage

https://github.com/user-attachments/assets/9a66a4e7-b151-42e9-aa9c-600596e17fed

To start your installed services:

```bash
cd mbox-uber/
./server.sh
```

## [FAQ](./docs/FAQ.md)

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{inproceedings,
author = {Li, Zaibei and Jensen, Martin and Nolte, Alexander and Spikol, Daniel},
year = {2024},
month = {03},
pages = {785-791},
title = {Field report for Platform mBox: Designing an Open MMLA Platform},
doi = {10.1145/3636555.3636872}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
