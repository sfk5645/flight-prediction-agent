{{ config(materialized='view') }}

-- Hourly Open-Meteo bronze → feature-friendly names (joined on date + hour).
select
  cast(date as date) as weather_date,
  cast(hour as integer) as weather_hour,
  upper(cast(airport as varchar)) as airport,
  cast(temperature_2m as double) as temperature_2m_mean,
  cast(precipitation as double) as precipitation_sum,
  cast(wind_speed_10m as double) as windspeed_10m_max,
  cast(weather_code as integer) as weathercode,
  cast(lat as double) as lat,
  cast(lon as double) as lon
from read_parquet(
  '{{ env_var("FLIGHT_PARQUET_ROOT") }}/weather/**/*.parquet',
  hive_partitioning=true,
  union_by_name=true
)
where date is not null
  and airport is not null
  and hour is not null
