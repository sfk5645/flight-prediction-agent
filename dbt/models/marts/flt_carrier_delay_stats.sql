{{ config(materialized='table') }}

select
  op_unique_carrier,
  count(*) as n_flights,
  avg(arr_delay) as avg_arr_delay,
  avg(arr_delay_15) as pct_delay_15,
  avg(dep_delay) as avg_dep_delay,
  avg(taxi_out) as avg_taxi_out,
  avg(nas_delay) as avg_nas_delay,
  avg(late_aircraft_delay) as avg_late_aircraft_delay
from {{ ref('flt_flights_clean') }}
group by 1
