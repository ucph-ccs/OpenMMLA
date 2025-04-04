# InfluxDB OSS 2.7.1

## Data Organizational Structure

```bash
Database Server (InfluxDB)
|         |
|         |--- Organization (name: group_01_db; username: user1; password: pwd1)
|         |    |
|         |    |--- Bucket (name: session_<session-start-time>)
|         |    |    |
|         |    |    |--- Measurement (type: speaker_recognition)
|         |    |    |    |
|         |    |    |    |--- Point
|         |    |    |    |    |
|         |    |    |    |    |--- Tag
|         |    |    |    |    |
|         |    |    |    |    |--- Field
|         |    |    |    |    |
|         |    |    |    |    |--- Timestamp
|         |    |    |
|         |    |    |--- Measurement (type: speech_transcription)
|         |    |    |--- Measurement (type: translation)
|         |    |    |--- Measurement ...
|         |    |
|         |    |--- Bucket ...
|         |
|         |--- Organization (name: group_02_db; username: user2; password: pwd2)
|
Client (Influx CLI)
```
<details>
<summary><strong>Definition</strong></summary>

+ Organization: An InfluxDB organization is a workspace for a group of users and acts as a storage for multiple buckets.
+ Bucket: A bucket is a named location where time series data are stored. It is capable of containing multiple measurements.
+ Measurement: This is a logical group for time series data. All points within a given measurement share the same set of tags. Each measurement encompasses numerous tags and fields and corresponds to a specific basestation.
+ Tags: These are key-value pairs that exhibit infrequent changes. They are intended to store metadata for each point, providing identifiers for the data source such as host, location, station, etc.
+ Fields: Fields are key-value pairs where values fluctuate over time. These include metrics like temperature, pressure, stock price, etc.
+ Timestamp: This represents the specific time associated with the data. When data is stored and queried, it is sorted according to the timestamp.
+ Point: A point is a singular data record that is identified by its measurement, tag keys, tag values, field key, and timestamp.
+ Series: A series comprises a group of points sharing the same measurement, tag keys, and tag values.

</details>

## Setup InfluxDB

<details>
<summary>macOS</summary>

  + Install InfluxDB && InfluxDB CLI (needs [homebrew](https://brew.sh/))
    ```cmd
    brew install influxdb
    brew install influxdb-cli
    ```
    
  + Start InfluxDB server
    ```cmd
    brew services start influxdb
    ```
  
  + Go to `localhost:8086`, and follow the instructions to create admin user with operator API token, save your token
    in a safe place.

</details>

<details>
<summary>Ubuntu</summary>

  + Install InfluxDB & InfluxDB CLI
    ```cmd
    # influxdata-archive_compat.key GPG fingerprint:
    #     9D53 9D90 D332 8DC7 D6C8 D3B9 D8FF 8E1F 7DF8 B07E
    wget -q https://repos.influxdata.com/influxdata-archive_compat.key
    echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
    echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
        
    sudo apt-get update && sudo apt-get install influxdb2
    sudo apt-get update && sudo apt-get install influxdb2-cli
    ```
      
  + Start InfluxDB server
    ```cmd
    sudo service influxdb start
    ```
    
  + Go to `localhost:8086`, and follow the instructions to create admin user with operator API token, save your token in a safe place.
  
</details>


## Miscellaneous (Optional)

### Config InfluxDB CLI 
+ Create a connection configuration and set it active
  ```cmd
  influx config create --active \
  -n config-name \
  -u http://localhost:8086 \
  -t API-TOKEN \
  -o example-org
  ```
+ Check config file
  ```cmd
  cat ~/.influxdbv2/conf
  ```
  
### CLI User management
+ Creat orgnization
  ```cmd
  influx org create -n [org-name]
  ```

+ Create user
  ```cmd
  influx user create -n [usr-name] -p [usr-pwd] -o [usr-org]
  ```

+ Create authorization
  ```cmd
  # grant all access in a single organization
  influx auth create -u [usr-name] --all-access -o [org-name]
  # grant all access to all organization
  influx auth create -u [usr-name] --operator

### Reset InfluxDB server
<details>
<summary>Mac</summary>

  ```cmd
  # stop the InfluxDB server
  brew services stop influxdb
  
  # delete the whole database and config file
  rm -rf ~/.influxdbv2
  
  # restart server
  brew services start influxdb
  ```

+ Go to `localhost:8086` in browser, reset the admin user

</details>

<details>
<summary>Linux</summary>

  ```cmd
  # Stop the InfluxDB server
  sudo service influxdb stop
  
  # delete the whole database and config file
  sudo rm -rf /var/lib/influxdb/
  sudo rm -rf /etc/influxdb/
  
  # restart the InfluxDB server
  sudo service influxdb start
  ```

+ Go to `localhost:8086`, reset the admin user

</details>