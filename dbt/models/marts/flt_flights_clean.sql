{{ config(materialized='table') }}

with base as (
  select
    f.*,
    case when coalesce(f.cancelled, 0) = 0 and f.arr_delay >= 15 then 1 else 0 end as arr_delay_15,
    case when coalesce(f.cancelled, 0) = 0 and f.dep_delay >= 15 then 1 else 0 end as dep_delay_15,
    extract(month from f.fl_date) as fl_month,
    extract(dow from f.fl_date) as fl_dow,
    case
      when f.crs_dep_time is null then null
      else cast(floor(f.crs_dep_time / 100) as integer)
    end as crs_dep_hour,
    case
      when f.crs_arr_time is null then null
      else cast(floor(f.crs_arr_time / 100) as integer)
    end as crs_arr_hour,
    case
      when extract(dow from f.fl_date) in (0, 6) then 1
      else 0
    end as is_weekend,
    case
      when cast(floor(f.crs_dep_time / 100) as integer) between 6 and 9 then 1
      when cast(floor(f.crs_dep_time / 100) as integer) between 16 and 20 then 1
      else 0
    end as is_peak_hour
  from {{ ref('stg_flights') }} f
  where coalesce(f.cancelled, 0) = 0
    and coalesce(f.diverted, 0) = 0
    and f.arr_delay is not null
),

with_volume as (
  select
    b.*,
    -- Same-day scheduled bank size (known from schedule; congestion proxy)
    count(*) over (
      partition by b.origin, b.fl_date, b.crs_dep_hour
    ) as origin_hour_sched_flights,
    count(*) over (
      partition by b.dest, b.fl_date, b.crs_arr_hour
    ) as dest_hour_sched_flights,
    count(*) over (
      partition by b.origin, b.fl_date
    ) as origin_day_sched_flights
  from base b
)

select * from with_volume
