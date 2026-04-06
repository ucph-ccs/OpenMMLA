#!/usr/bin/env python3
"""migrate data from old per-session InfluxDB buckets to the new single-bucket schema.

reads all data from old buckets (6 separate measurements) and rewrites them into
a single 'mmla-data' bucket using the new sensor_events measurement with
session_id + event_type tags.

usage:
    python scripts/migrate_influxdb.py -c <config_path>

the config file should have the new InfluxDB settings (with 'bucket: mmla-data').
the script will enumerate old buckets from InfluxDB and migrate each one.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import yaml


OLD_MEASUREMENT_TO_EVENT_TYPE = {
    "speaker_transcription": "asr_transcription",
    "speaker_recognition": "asr_recognition",
    "badge_translation": "ips_translation",
    "badge_rotation": "ips_rotation",
    "badge_relation": "ips_relation",
    "action_recognition": "vfa_action",
}

SYSTEM_BUCKETS = {"_tasks", "_monitoring"}
NEW_MEASUREMENT = "sensor_events"


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate old per-session InfluxDB buckets to new schema")
    parser.add_argument("-c", "--config", required=True, help="path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="print what would be migrated without writing")
    return parser.parse_args()


def get_old_buckets(client):
    """list all user-created buckets (potential old session buckets)."""
    buckets_api = client.buckets_api()
    all_buckets = buckets_api.find_buckets()
    return [b.name for b in all_buckets.buckets if b.name not in SYSTEM_BUCKETS]


def query_old_measurement(client, org, bucket_name, measurement):
    """query all data from an old measurement in a bucket."""
    query_api = client.query_api()

    # try to extract start time from bucket name
    start = "-365d"
    try:
        last_seg = bucket_name.rsplit('_', 1)[-1]
        dt = datetime.strptime(last_seg, '%y%m%dT%H%MZ')
        start = dt.replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        try:
            timestamp_str = bucket_name.split('_', 1)[1]
            dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%SZ')
            start = dt.replace(tzinfo=timezone.utc).isoformat()
        except (IndexError, ValueError):
            pass

    query = f'''
        from(bucket: "{bucket_name}")
        |> range(start: {start})
        |> filter(fn: (r) => r._measurement == "{measurement}")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    try:
        return query_api.query(org=org, query=query)
    except Exception as e:
        print(f"  warning: query failed for {bucket_name}/{measurement}: {e}")
        return []


def migrate_bucket(client, org, target_bucket, old_bucket, dry_run=False):
    """migrate all measurements from an old bucket to the new schema."""
    write_api = client.write_api(write_options=SYNCHRONOUS) if not dry_run else None
    total_points = 0

    for old_measurement, event_type in OLD_MEASUREMENT_TO_EVENT_TYPE.items():
        tables = query_old_measurement(client, org, old_bucket, old_measurement)

        count = 0
        for table in tables:
            for record in table.records:
                point = Point(NEW_MEASUREMENT)
                point = point.tag("session_id", old_bucket)
                point = point.tag("event_type", event_type)

                for key, value in record.values.items():
                    if key in ("_start", "_stop", "_time", "_measurement",
                               "result", "table", "_field", "_value"):
                        continue
                    if value is not None:
                        if isinstance(value, (int, float)):
                            point = point.field(key, float(value))
                        else:
                            point = point.field(key, str(value))

                point = point.time(record.get_time(), WritePrecision.NS)

                if not dry_run:
                    write_api.write(bucket=target_bucket, org=org, record=point)
                count += 1

        if count > 0:
            print(f"  {old_measurement} -> {event_type}: {count} points")
            total_points += count

    return total_points


def main():
    args = parse_args()

    config = yaml.safe_load(open(args.config, 'r'))
    influx_config = config['InfluxDB']
    url = influx_config['url']
    token = influx_config['token']
    org = influx_config['org']
    target_bucket = influx_config.get('bucket', 'mmla-data')

    client = InfluxDBClient(url=url, token=token, org=org)

    old_buckets = get_old_buckets(client)
    # filter out the target bucket itself
    old_buckets = [b for b in old_buckets if b != target_bucket]

    if not old_buckets:
        print("No old session buckets found to migrate.")
        sys.exit(0)

    print(f"Found {len(old_buckets)} old bucket(s) to migrate:")
    for b in old_buckets:
        print(f"  - {b}")
    print(f"Target bucket: {target_bucket}")
    print(f"Dry run: {args.dry_run}")
    print()

    if not args.dry_run:
        confirm = input("Proceed with migration? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Migration cancelled.")
            sys.exit(0)

    grand_total = 0
    for old_bucket in old_buckets:
        print(f"\nMigrating: {old_bucket}")
        count = migrate_bucket(client, org, target_bucket, old_bucket, dry_run=args.dry_run)
        grand_total += count
        if count == 0:
            print("  (no data found)")

    print(f"\nMigration complete. Total points migrated: {grand_total}")
    if not args.dry_run:
        print("Old buckets were NOT deleted. Delete them manually after verifying the migration.")

    client.close()


if __name__ == "__main__":
    main()
