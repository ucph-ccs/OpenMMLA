# Database Setup

OpenMMLA uses two databases:
- **InfluxDB** (time series): stores real-time sensor event data (measurements, tags, fields with timestamps)
- **MongoDB** (document): stores session metadata, experiment configurations, and analytics results

## InfluxDB OSS 2.7

### Data Organizational Structure

OpenMMLA uses a single-bucket, single-measurement schema. All pipeline data is written to one measurement and differentiated by `session_id` and `event_type` tags.

```
Organization
|
|--- Bucket (default: "mmla-data")
|    |
|    |--- Measurement: "sensor_events"
|         |
|         |--- Point
|         |    |--- Tags:  session_id, event_type
|         |    |--- Fields: (varies by event_type)
|         |    |--- Timestamp
|         |
|         |--- Event Types:
|              |
|              |--- asr_transcription
|              |    Fields: window_start_time, window_end_time, text, words, speaker
|              |
|              |--- asr_recognition
|              |    Fields: window_start_time, window_end_time, speakers, similarities,
|              |            durations, segment_start_times
|              |
|              |--- ips_translation
|              |    Fields: window_start_time, window_end_time, translations
|              |
|              |--- ips_rotation
|              |    Fields: window_start_time, window_end_time, rotations
|              |
|              |--- ips_relation
|              |    Fields: window_start_time, window_end_time, graph
|              |
|              |--- vfa_action
|                   Fields: window_start_time, window_end_time, action_recognition
```

<details>
<summary><strong>Definition</strong></summary>

+ Organization: An InfluxDB organization is a workspace for a group of users.
+ Bucket: A named location where time series data are stored. OpenMMLA uses a single bucket (`mmla-data` by default) for all sessions.
+ Measurement: A logical group for time series data. OpenMMLA uses a single measurement (`sensor_events`) for all event types.
+ Tags: Key-value pairs for indexing and filtering. OpenMMLA uses `session_id` (to identify the recording session) and `event_type` (to distinguish ASR/IPS/VFA data).
+ Fields: Key-value pairs where values fluctuate over time (e.g., transcription text, speaker embeddings, position coordinates).
+ Timestamp: The time associated with each data point, used for sorting and range queries.
+ Point: A singular data record identified by its measurement, tags, fields, and timestamp.

</details>

### Setup InfluxDB
```bash
# For macOS
brew install influxdb influxdb-cli
brew services start influxdb

# For Ubuntu/Debian
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
sudo apt update && sudo apt install influxdb2 influxdb2-cli
sudo systemctl enable influxdb
sudo systemctl start influxdb

# Go to http://localhost:8086, and follow the instructions to create admin user with operator API token, save your token in a safe place, 
# it will be used for config later.
```

### Miscellaneous

#### Config InfluxDB CLI 
+ Create a connection configuration and set it active
  ```bash
  influx config create --active \
  -n config-name \
  -u http://localhost:8086 \
  -t API-TOKEN \
  -o example-org
  ```
+ Check config file
  ```bash
  cat ~/.influxdbv2/conf
  ```
  
#### CLI User management
+ Create organization
  ```bash
  influx org create -n [org-name]
  ```

+ Create user
  ```bash
  influx user create -n [usr-name] -p [usr-pwd] -o [usr-org]
  ```

+ Create authorization
  ```bash
  # grant all access in a single organization
  influx auth create -u [usr-name] --all-access -o [org-name]
  # grant all access to all organization
  influx auth create -u [usr-name] --operator
  ```

#### Reset InfluxDB server
<details>
<summary>macOS</summary>

  ```bash
  brew services stop influxdb
  rm -rf ~/.influxdbv2
  brew services start influxdb
  ```

+ Go to `localhost:8086` in browser, reset the admin user

</details>

<details>
<summary>Linux</summary>

  ```bash
  sudo service influxdb stop
  sudo rm -rf /var/lib/influxdb/
  sudo rm -rf /etc/influxdb/
  sudo service influxdb start
  ```

+ Go to `localhost:8086`, reset the admin user

</details>

---

## MongoDB

### Data Organizational Structure

```bash
Database Server (MongoDB)
|         |
|         |--- Database (name: openmmla)
|         |    |
|         |    |--- Collection (name: sessions)
|         |    |    |
|         |    |    |--- Document { session_id, start_time, end_time, group, ... }
|         |    |
|         |    |--- Collection (name: experiments)
|         |    |    |
|         |    |    |--- Document { experiment_id, name, config, ... }
|         |    |
|         |    |--- Collection ...
```

<details>
<summary><strong>Definition</strong></summary>

+ Database: A MongoDB database holds collections of documents. OpenMMLA uses the `openmmla` database by default.
+ Collection: A collection is a grouping of documents, analogous to a table in relational databases. Documents within a collection can have different fields.
+ Document: A document is a record in a collection, stored as BSON (binary JSON). Each document has a unique `_id` field.

</details>

### Setup MongoDB

```bash
# For macOS (via Homebrew)
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# For Ubuntu/Debian
# Import the public key
curl -fsSL https://www.mongodb.org/static/pgp/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
sudo apt update && sudo apt install -y mongodb-org
sudo systemctl enable mongod
sudo systemctl start mongod

# Verify: connect to MongoDB shell
mongosh
```

### Miscellaneous

#### Connect with mongosh
```bash
# connect to local instance (default port 27017)
mongosh

# connect to a specific host/port
mongosh "mongodb://hostname:27017"
```

#### Basic operations
```bash
# show databases
show dbs

# switch to openmmla database
use openmmla

# show collections
show collections

# query documents in a collection
db.sessions.find().pretty()
```

#### Reset MongoDB server
<details>
<summary>macOS</summary>

  ```bash
  brew services stop mongodb-community
  rm -rf /opt/homebrew/var/mongodb/*
  brew services start mongodb-community
  ```

</details>

<details>
<summary>Linux</summary>

  ```bash
  sudo systemctl stop mongod
  sudo rm -rf /var/lib/mongodb/*
  sudo systemctl start mongod
  ```

</details>
