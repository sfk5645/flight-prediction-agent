{{ config(materialized='table') }}

-- Historical congestion / delay-cause profiles by airport and clock hour.
-- Used as lookup features (not same-flight taxi/NAS outcomes) to limit leakage.

with departures as (
  select
    origin as airport,
    crs_dep_hour as hour,
    taxi_out,
    nas_delay,
    carrier_delay,
    weather_delay,
    late_aircraft_delay,
    security_delay,
    dep_delay,
    arr_delay_15
  from {{ ref('flt_flights_clean') }}
  where crs_dep_hour is not null
),

arrivals as (
  select
    dest as airport,
    crs_arr_hour as hour,
    taxi_in,
    arr_delay_15
  from {{ ref('flt_flights_clean') }}
  where crs_arr_hour is not null
),

dep_agg as (
  select
    airport,
    hour,
    count(*) as n_departures,
    avg(taxi_out) as avg_taxi_out,
    avg(nas_delay) as avg_nas_delay,
    avg(carrier_delay) as avg_carrier_delay,
    avg(weather_delay) as avg_weather_delay,
    avg(late_aircraft_delay) as avg_late_aircraft_delay,
    avg(security_delay) as avg_security_delay,
    avg(dep_delay) as avg_dep_delay,
    avg(arr_delay_15) as pct_arr_delay_15_from_deps
  from departures
  group by 1, 2
),

arr_agg as (
  select
    airport,
    hour,
    count(*) as n_arrivals,
    avg(taxi_in) as avg_taxi_in,
    avg(arr_delay_15) as pct_arr_delay_15_from_arrs
  from arrivals
  group by 1, 2
)

select
  coalesce(d.airport, a.airport) as airport,
  coalesce(d.hour, a.hour) as hour,
  coalesce(d.n_departures, 0) as n_departures,
  coalesce(a.n_arrivals, 0) as n_arrivals,
  coalesce(d.n_departures, 0) + coalesce(a.n_arrivals, 0) as n_operations,
  d.avg_taxi_out,
  a.avg_taxi_in,
  d.avg_nas_delay,
  d.avg_carrier_delay,
  d.avg_weather_delay,
  d.avg_late_aircraft_delay,
  d.avg_security_delay,
  d.avg_dep_delay,
  coalesce(d.pct_arr_delay_15_from_deps, a.pct_arr_delay_15_from_arrs) as pct_delay_15
from dep_agg d
full outer join arr_agg a
  on d.airport = a.airport
 and d.hour = a.hour
