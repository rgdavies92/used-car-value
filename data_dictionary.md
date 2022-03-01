| Column Name | Description | Type | Count | Used in Modelling  |     
|  ------  | -----|----|----|-----|
| name        | Short car name | str | 400247 non-null | No |
| name_subtitle | Long car name | str | 400247 non-null | No | 
| year        | Year made | int64 | 400247 non-null | Yes |
| price       | AutoTrader asking price (£) | float64 | 400247 non-null | Target for decision tree models |
| log_price   | Car log-price for linear models | float64         | 400247 non-null  | Target for linear models | 
| body        | Car body type - one of nine AutoTrader defined options | str | 400247 non-null | Yes | 
| mileage     | Car mileage | float64 | 400247 non-null | No |
| log_mileage | Car log-mileage | float64 | 400247 non-null | Yes |
| BHP         | Car brake horse power | float64 | 400247 non-null | Yes |
| doors       | Number of doors | str || 400247 non-null | Yes | 
| transmission | Automatic or manual gearbox | str | 400247 non-null | Yes | 
| make        | Car brand | str | 400247 non-null | Yes | 
| fuel        | Car fuel option - one of eight Autotrader defined options | str | 400247 non-null | Yes | 
| mpg         | Car miles per gallon - one of five AutoTrader defined options | str | 400247 non-null | No |
| drivetrain  | How car transmission is connected to axles | str | 400247 non-null | Yes |
| engine      | Engine size in litres for non-electric cars | float64 | 392743 non-null | No |
| owners      | Number of previous owners | float64 | 199447 non-null | No |
| ULEZ        | Flag for Ultra Low Emission Zone compliance | str | 352090 non-null | No |
| county      | Dealer county from GeoPy | str | 398697 non-null | No | 
| dealer_area | Dealer area from scraped href | str        | 340545 non-null | No | 
| dealer_city | Dealer city from scraped href | str        | 399771 non-null | No | 
| dealer_lat  | Dealer latitude from GeoPy | float64         | 398697 non-null | No | 
| dealer_lon  | Dealer longitude from GeoPy | float64          | 398697 non-null | No | 
| geocode     | Raw Geocode return from reverse GeoPy search | str           | 398697 non-null | No | 
| postcode    | Dealer postcode | str           | 398697 non-null | No | 
| postcode_short | Dealer postcode shortened to outcode | str     | 386214 non-null  | No | 
| seller1     | Dealer rating | float64        | 329526 non-null  | No | 
| used        | Used/New flag. 1 is used. 0 is new. | int64        | 400247 non-null  | No |   
| e_engine_kW | Engine Size in kW for electric cars | float64        | 5343 non-null    | No | 
| log_price   | Car log-price for linear models | float64         | 400247 non-null  | Only linear models | 
| orig_name   | Original car name | str        | 400247 non-null  | No |  
| id          | AutoTrader car ID | int64        | 400247 non-null  | No |   
| year_reg    | Car year and corresponding registration | str        | 373067 non-null  | No |  
| link        | AutoTrader listing URL | str        | 400247 non-null  | No |  
| href0       | Dealer href | str        | 340545 non-null  | No |  
