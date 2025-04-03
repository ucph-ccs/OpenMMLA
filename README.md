# OpenMMLA

OpenMMLA a toolkit for multimodal learning analytics, providing various built-in pipelines for different tasks. The toolkit is for building up the MMLA pipeline as shown below:

<img src="docs/high-level-system-design.png" alt="OpenMMLA system design" width="100%">

<details>
<summary><strong>High-level System Design Description</strong></summary>

High-level system design with data flow. The platform's high-level design consists of three stages: input, processing, and output. In the data input stage, multimodal raw data from sensors & wearable badges are streamed directly to base stations or a central media server. The raw inputs are transformed into structured, coded streams for efficient transmission and processing. In the data processing stage, these encoded streams are processed individually by the corresponding Base, which handles some signal processing, while more complex tasks are offloaded to the server. For Bases within the same group, results are synchronized and uploaded to the time series database, where segment-level measurement features are generated. Finally, in the data output stage, these measurement features are visualized on the dashboard in real time and combined into indicators to analyze group interactions. The platform also generates post-processing visualizations, logs, and reports, which are stored and accessible via the shared dashboard, enabling both real-time awareness and retrospective analysis of group dynamics.
</details>

## Quick Setup


### Pre-implemented Pipelines

+ [ASR with Diarization](base_stations/asr/README.md)
+ [Indoor Positioning System](base_stations/ips/README.md)
+ [Video Frame Analyzer](base_stations/vfa/README.md)
