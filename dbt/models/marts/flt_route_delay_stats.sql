{{ config(materialized='table') }}

select
  origin,
  dest,
  op_unique_carrier,
  count(*) as n_flights,
  avg(arr_delay) as avg_arr_delay,
  avg(arr_delay_15) as pct_delay_15,
  avg(distance) as avg_distance,
  avg(taxi_out) as avg_taxi_out,
  avg(taxi_in) as avg_taxi_in,
  avg(nas_delay) as avg_nas_delay,
  avg(carrier_delay) as avg_carrier_delay,
  avg(late_aircraft_delay) as avg_late_aircraft_delay,
  avg(crs_elapsed_time) as avg_crs_elapsed_time
from {{ ref('flt_flights_clean') }}
group by 1, 2, 3
