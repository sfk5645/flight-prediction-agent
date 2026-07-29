{{ config(materialized='view') }}

select
  cast(fl_date as date) as fl_date,
  upper(cast(op_unique_carrier as varchar)) as op_unique_carrier,
  cast(op_carrier_fl_num as integer) as op_carrier_fl_num,
  upper(cast(origin as varchar)) as origin,
  upper(cast(dest as varchar)) as dest,
  cast(crs_dep_time as integer) as crs_dep_time,
  cast(crs_arr_time as integer) as crs_arr_time,
  cast(dep_delay as double) as dep_delay,
  cast(arr_delay as double) as arr_delay,
  cast(cancelled as integer) as cancelled,
  cast(diverted as integer) as diverted,
  cast(distance as double) as distance,
  cast(taxi_out as double) as taxi_out,
  cast(taxi_in as double) as taxi_in,
  cast(crs_elapsed_time as double) as crs_elapsed_time,
  cast(actual_elapsed_time as double) as actual_elapsed_time,
  cast(air_time as double) as air_time,
  cast(carrier_delay as double) as carrier_delay,
  cast(weather_delay as double) as weather_delay,
  cast(nas_delay as double) as nas_delay,
  cast(security_delay as double) as security_delay,
  cast(late_aircraft_delay as double) as late_aircraft_delay
from read_parquet(
  '{{ env_var("FLIGHT_PARQUET_ROOT") }}/bts/**/*.parquet',
  hive_partitioning=true,
  union_by_name=true
)
where fl_date is not null
  and origin is not null
  and dest is not null
