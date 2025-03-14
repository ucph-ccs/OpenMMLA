# InfluxDB OSS 2.7.1

## Data Organizational Structure

```bash
InfluxDB Server BS
|         |
|         |--- Organization (name: group_01_db; username: user1; password: pwd1)
|         |    |
|         |    |--- Bucket (name: session_01)
|         |    |    |
|         |    |    |--- Measurement (type: audio)
|         |    |    |    |
|         |    |    |    |--- Point
|         |    |    |    |    |
|         |    |    |    |    |--- Tag
|         |    |    |    |    |
|         |    |    |    |    |--- Field
|         |    |    |    |    |
|         |    |    |    |    |--- Timestamp
|         |    |    |
|         |    |    |--- Measurement (type: video)
|         |    |    |--- Measurement (type: proximity)
|         |    |
|         |    |--- Bucket (name: session_02)
|         |
|         |--- Organization (name: group_02_db; username: user2; password: pwd2)
|
InfluxDB Client BS (audio, video, proximity)
```

### Definition
+ Organization: An InfluxDB organization is a workspace for a group of users and acts as a storage for multiple buckets.
+ Bucket: A bucket is a named location where time series data are stored. It is capable of containing multiple measurements.
+ Measurement: This is a logical group for time series data. All points within a given measurement share the same set of tags. Each measurement encompasses numerous tags and fields and corresponds to a specific basestation.
+ Tags: These are key-value pairs that exhibit infrequent changes. They are intended to store metadata for each point, providing identifiers for the data source such as host, location, station, etc.
+ Fields: Fields are key-value pairs where values fluctuate over time. These include metrics like temperature, pressure, stock price, etc.
+ Timestamp: This represents the specific time associated with the data. When data is stored and queried, it is sorted according to the timestamp.
+ Point: A point is a singular data record that is identified by its measurement, tag keys, tag values, field key, and timestamp.
+ Series: A series comprises a group of points sharing the same measurement, tag keys, and tag values.

## Installation
+ Windows
  + Install InfluxDB
    ```cmd
    Expand-Archive .\influxdb2-2.7.0-arduino-amd64.zip -DestinationPath 'C:\Program Files\InfluxData\'
    ```
    Rename `C:\Program Files\InfluxData\influxdb` and set the directory to system environment variables

  + Install Influx CLI
    ```cmd
    Expand-Archive .\influxdb2-client-2.7.3-arduino-amd64.zip -DestinationPath 'C:\Program Files\InfluxData\'
    ```
    Rename to `C:\Program Files\InfluxData\influx` and set the directory to system environment variables

  + Start InfluxDB server
    ```cmd
    influxd
    ```
  
  + Go to `localhost:8086`, and follow the instructions to create admin user with operator API token

+ Mac
  + Install InfluxDB
    ```cmd
    brew update
    brew install influxdb
    ```
  
  + Install Influx CLI
    ```cmd
    brew install 'influxdb-cli'
    ```
  
  + Start InfluxDB server
    ```cmd
    influxd
    ```

  + Go to `localhost:8086`, and follow the instructions to create admin user with operator API token
    
+ Raspberry Pi
  + Download InfluxDB DEB file
    ```cmd
    wget -q https://repos.influxdata.com/influxdata-archive_compat.key
    echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
    echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list
    
    sudo apt-get update && sudo apt-get install influxdb2
    ```
  
  + Install InfluxDB
    ```cmd
    sudo dpkg -i influxdb2-2.7.1-arm64.deb
    ```
    
  + Install Influx CLI
    + Download package
      ```cmd
      wget https://dl.influxdata.com/influxdb/releases/influxdb2-client-2.7.3-linux-arm64.tar.gz
      ```
    + Unpackage the downloaded package
      ```cmd
      tar xvzf [path/to/influxdb2-client-2.7.3-linux-arm64.tar.gz]
      ```
    + Place the unpackaged influx executable in your system $PATH
      ```cmd
      sudo cp influxdb2-client-2.7.3-linux-arm64/influx /usr/local/bin/
      ```
    
  + Start InfluxDB server
    ```cmd
    sudo service influxdb start
    ```
    
  + Go to `localhost:8086`, and follow the instructions to create admin user with operator API token
  
## Configuration

### Admin 
  - Username: admin
  - Password: 12345678
  - Organization Name: admin
  - Bucket Name: e.g. session_01

### InfluxDB CLI Configuration Setup
+ Initialize with operator API token
  ```cmd
  influx config create --config-name [admin] --host-url [http://localhost:8086]--org [admin] --token [Operator API Token] --active
  ```
+ Check config file
  
  ```cmd
  # Windows
  cat ~\.influxdbv2\configs
  # Mac
  cat ~/.influxdbv2/conf
  # Raspberry Pi
  cat /var/lib/influxdb/conf
  ```

### Export operator API token

+ Windows
  Go to control panel -> searh 'environment variables' -> Advanced -> Environment Variables -> System Variables -> new ->
  ```
  Name: INFLUXDB_TOKEN 
  Value: 64OWQoO9rFFdlpDGqtFHJJ6790-mHv0oXbX2n8Ig0Tas5mFdQlQSezAFcXLpeecCelTfeJ1HKCuJGUfm1qYhNw==
  ```

+ Mac
  + Open bash 
    ```cmd
    nano ~/.bash_profile
    ```
  + Add line
    ```cmd
    export INFLUXDB_TOKEN='hR5q4Waye8nh2mjUsYup-1n7j0aarrqcnmJIVDs6Z0RAwlMVwOpt9gPcenzvDvlV1FN1dh3LXkLTyYTApTf7pg=='
    ```
  + Reload bash
    ```cmd
    source ~/.bash_profile
    ```

+ Raspberry Pi
  + Open bash
    ```cmd
    nano ~/.bashrc
    ```
  + Add line
    ```cmd
    export INFLUXDB_TOKEN='mLbdhupqgFApa1BTpYur8dgLjSG9nwZKrx-_R0db-ct5N2JViYvc8oUG4TDJNrI8Gn0GoEvM5xFYucaZHVTOog=='
    ```

## Python API
### Install influxdb_client
```cmd
pip install 'influxdb-client[ciso]'
```
```python
# initialize InfluxDB
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
 
token = os.environ.get("INFLUXDB_TOKEN")
org = "MAS"
url = "http://localhost:8086"
write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

# write data
bucket="MAS" # your bucket name

write_api = write_client.write_api(write_options=SYNCHRONOUS)
   
for value in range(5):
  point = (
    Point("measurement1")
    .tag("tagname1", "tagvalue1")
    .field("field1", value)
  )
  write_api.write(bucket=bucket, org="MAS", record=point)
  time.sleep(1) # separate points by 1 second
  
# read data
query_api = write_client.query_api()

query = """from(bucket: "MAS")
 |> range(start: -10m)
 |> filter(fn: (r) => r._measurement == "measurement1")"""
tables = query_api.query(query, org="MAS")

for table in tables:
  for record in table.records:
    print(record)
    
# calculate mean value   
query = """from(bucket: "MAS")
  |> range(start: -10m)
  |> filter(fn: (r) => r._measurement == "measurement1")
  |> mean()"""
tables = query_api.query(query, org="MAS")

for table in tables:
    for record in table.records:
        print(record)
```

## Miscellaneous

### Reset InfluxDB server
  + Windows
    + Delete the whole database and config file
      ```
      %USERPROFILE%\.influxdb
      ```
    + Restart the InfluxDB server
      ```cmd
      influxd
      ```
    + Go to `localhost:8086`, reset the admin user

  + Mac
    + Delete the whole database and config file
      ```cmd
      rm -rf ~/.influxdbv2
      ```
    + Restart the InfluxDB server
      ```cmd
      influxd
      ```
    + Go to `localhost:8086`, reset the admin user

  + Raspberry Pi
    + Stop the InfluxDB server
      ```cmd
      sudo service influxdb stop
      ```
    + Delete the whole database and config file
      ```cmd
      sudo rm -rf /var/lib/influxdb/
      sudo rm -rf /etc/influxdb/
      ```
    + Restart the InfluxDB server
      ```cmd
      sudo service influxdb start
      ```
    + Go to `localhost:8086`, reset the admin user

### User management
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
  ```
