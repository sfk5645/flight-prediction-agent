{{ config(materialized='table') }}

-- Feature-ready flight grain: hourly weather + historical congestion/carrier profiles.
-- Origin weather joined on CRS departure hour; dest on CRS arrival hour.

select
  f.*,
  w.temperature_2m_mean as origin_temp_c,
  w.precipitation_sum as origin_precip_mm,
  w.windspeed_10m_max as origin_wind_kmh,
  w.weathercode as origin_weathercode,
  wd.temperature_2m_mean as dest_temp_c,
  wd.precipitation_sum as dest_precip_mm,
  wd.windspeed_10m_max as dest_wind_kmh,
  wd.weathercode as dest_weathercode,
  -- Origin departure-hour congestion profile
  oh.avg_taxi_out as origin_hist_avg_taxi_out,
  oh.avg_nas_delay as origin_hist_avg_nas_delay,
  oh.avg_carrier_delay as origin_hist_avg_carrier_delay,
  oh.avg_weather_delay as origin_hist_avg_weather_delay,
  oh.avg_late_aircraft_delay as origin_hist_avg_late_aircraft_delay,
  oh.n_operations as origin_hist_hour_ops,
  oh.pct_delay_15 as origin_hist_hour_pct_delay_15,
  -- Dest arrival-hour congestion profile
  dh.avg_taxi_in as dest_hist_avg_taxi_in,
  dh.n_operations as dest_hist_hour_ops,
  dh.pct_delay_15 as dest_hist_hour_pct_delay_15,
  -- Carrier reliability
  c.pct_delay_15 as carrier_hist_pct_delay_15,
  c.avg_taxi_out as carrier_hist_avg_taxi_out,
  c.avg_late_aircraft_delay as carrier_hist_avg_late_aircraft_delay
from {{ ref('flt_flights_clean') }} f
left join {{ ref('stg_weather') }} w
  on f.origin = w.airport
 and f.fl_date = w.weather_date
 and f.crs_dep_hour = w.weather_hour
left join {{ ref('stg_weather') }} wd
  on f.dest = wd.airport
 and f.fl_date = wd.weather_date
 and f.crs_arr_hour = wd.weather_hour
left join {{ ref('flt_airport_hour_stats') }} oh
  on f.origin = oh.airport
 and f.crs_dep_hour = oh.hour
left join {{ ref('flt_airport_hour_stats') }} dh
  on f.dest = dh.airport
 and f.crs_arr_hour = dh.hour
left join {{ ref('flt_carrier_delay_stats') }} c
  on f.op_unique_carrier = c.op_unique_carrier
