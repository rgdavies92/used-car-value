<p align="center">
  <img src="https://user-images.githubusercontent.com/69991618/111783988-c2483380-88b2-11eb-806b-6e2bbd495298.png" width="100"/> 
</p>

# Understanding Value in the UK Used Car Market in 2022

# Context

This project was completed as part of the General Assembly Data Science Immersive bootcamp. This document discusses the problem, hypothesis, methodology, conclusion, and tools used.

# Table of Contents

# Background and Problem Statement

The second-hand car market in the UK is currently thriving due to the shortage of new models and cars under three years of age. According to [some figures](https://www.bbc.com/news/business-58150025) the used car market has grown astronomically since the pandemic began, with more than 2.2 million used cars exchanged since 2019.  

A global shortage of computer chips used in car production, as well as other materials such as copper, aluminium and cobalt, has led to fewer new vehicles rolling off production lines. That has meant more buyers turning to the used-car market. Of course the car industry has not been the only industry to suffer from the global chip shortage, but despite popular demand, the car industry has struggled to deliver brand new cars to buyers. Computer chips are used in everything from modern infotainment systems, windscreen wipers and electric car batteries to name a few.

It’s estimated the chip crisis will cause 100,000 vehicles to not be delivered before the end of 2021, representing a huge blow for the industry. 

<b>Given that the chip shortage is unlikely to be resolved in the immediate future, now more than ever it would be helpful to be able to identify good value in the UK used car market. That is what this project intends to do.</b>

# Objectives

My primary objective for this project is to:

* Generate a predictive model for used car price in the UK in 2022.
* The final model must be interpretable.

My secondary objective is to test the hypothesis that:

* When all other car attributes are equal, A Dacia branded car is cheaper than a Volvo branded car.

# Data Acquisition

The data for this project came from [autotrader.co.uk](https://www.autotrader.co.uk/). Although AutoTrader do have an API, access is only permitted with a commercial contract. As such, the project data had to be obtained through use of a web scraper. Figure 1 below describes a typical search result on autotrader.co.uk

<br>
<p align="center" width="100%">
<kbd><img src="images/autotrader.png" width="700"  /></kbd>
</p>
<p align="center"><i><sub><b>Figure 1:</b> Sample autotrader.co.uk search result page with scraped fields highlighted in green.</sub></i></p>
<br>

Evidently, each car listing is rich in information. At this early stage in the project a 'more is more' approach was adopted. It was intended to gather all information in as few web scrapes as possible, without duplicating work by going back to revisit a car listing at a later date. Of particular interest was the free-form text box between car year and car price - often where car listings were lacking in specific details such as BHP, engine size or number of doors, they could be found and recovered from this string using RegEx. 

400,247 new and used cars were successfully scraped from AutoTrader. 

Two problems encoutered during web scraping:
* Problem 1 involved a hard limit on the search results. Search results were limited to 1000 car listings per search. This is not a problem if the search criteria are highly specific as the platform would expect, although for this project it was important to return all car listings. To navigate this problem, thousands of complimentary searches were performed and the number of returned results were verified to be less than 1000 before continuing to scrape the data.
* The solution to problem 1 likely contributed to problem 2; The searching behaviour was being flagged as problematic by the website. AutoTrader employ Cloudflare website security to prevent algorithms like this overloading their server with requests. The security successfuly prevented the Python Requests package from gaining access to search results, however switching to Cloudscraper instead resolved all issues. 

After a small number of iterative improvements to the web scraping function, the dataset was scraped in four days in the week commencing 31/01/2022. This is the date at which data are accurate.

<br>
<p align="center" width="100%">
<kbd><img src="images/predictors.png" width="500" /></kbd>
</p>
<p align="center"><i><sub><b>Figure 2:</b> Exhaustive list of the data points scraped for each car listing. Not all of them persisted into the final model, but that wasn't to be known at this stage.</sub></i></p>
<br>

# Data Cleaning and EDA

```python3
# Iterate over missing engine rows and use RegEx on name_subtitle to extract engine 
# size where possible. 

for index, car in ucars[ucars['engine'].isnull()].iterrows():
    car_subname = ucars.loc[index, 'name_subtitle']
    try:
        enginesize = re.findall('([0-9][.][0-9]+)',car_subname)[0]
    except: 
        enginesize = np.nan
    ucars.loc[index,'engine'] = float(enginesize)

# Remove L from each engine size.
ucars.engine= ucars.engine.apply(lambda x: float(str(x).replace('L','')))

# Drop remaining 105 used cars without engine size.
ucars.dropna(subset=['engine'],inplace=True)
```
</p>
<p align="center"><i><sub><b>Figure 3:</b> Sample codeblock using RegEx to extract missing enginesize data.</sub></i></p>
<br>

Data dictionary

Plots

# Modelling

# Evaluation

# Findings

# Limitations

# Conclusions

# Further Work

# Key Learnings and Challenges

# Libraries Used

# Contact
