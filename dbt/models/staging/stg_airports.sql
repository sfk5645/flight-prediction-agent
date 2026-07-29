{{ config(materialized='view') }}

select
  upper(cast(airport as varchar)) as airport,
  cast(airport_name as varchar) as airport_name,
  cast(lat as double) as lat,
  cast(lon as double) as lon,
  cast(city as varchar) as city,
  cast(country as varchar) as country,
  cast(type as varchar) as airport_type
from read_parquet(
  '{{ env_var("FLIGHT_PARQUET_ROOT") }}/airports/*.parquet',
  union_by_name=true
)
where airport is not null
