{{ config(materialized='view') }}

select
  cast(date as date) as weather_date,
  upper(cast(airport as varchar)) as airport,
  cast(temperature_2m_mean as double) as temperature_2m_mean,
  cast(precipitation_sum as double) as precipitation_sum,
  cast(windspeed_10m_max as double) as windspeed_10m_max,
  cast(weathercode as integer) as weathercode,
  cast(lat as double) as lat,
  cast(lon as double) as lon
from read_parquet(
  '{{ env_var("FLIGHT_PARQUET_ROOT") }}/weather/**/*.parquet',
  hive_partitioning=true,
  union_by_name=true
)
where date is not null
  and airport is not null
