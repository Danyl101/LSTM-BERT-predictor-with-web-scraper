                                    POST CONSTRUCTION MODEL LOG


Tags:[LSTM,TCN,SCRAPER,AUTOMATION,LOADER]
___________________________

Default Model [TCN]

So the initial issue was validation loss was too high ,training loss(0.8) validation loss(3.1) ,and final MSE value was around (8) , this meant that the model was not generalising properly or that validation and test datasets had wildy varying values compared to training dataset , when manually reviewed this seemed to be the case , as the validation and test values had around 2 point difference to training values , training value(-1.2) , validation value (1.8) test value (2.4), this was deduced to be due to the fact that nifty 50 is a rapidly growing indice and training values were acquired from a slow bullish-bearish market and covid collapse , while validation was taken from a strong bullish market and test was taken from very strong bullish market

Training Loss(0.8)
Validation Loss(3.1)
MSE(8.3)

___________________________

Iteration 1 [Split the datasets in preprocessing] [TCN]

Initially the dataset was scaled in group and then split in the model program , this would mean that even though it increases accuracy a bit  (as validation and test values are reduced to match training to a degree) , it would not hold in real life as they all should be treated as independent entities to simulate real life market movements , 
the splitting caused even a sharper drop with validation loss and mse , with training loss being (0.8) validation loss being (6.2) and mse being (25) 

Training Loss(0.9)
Validation Loss(6.3)
MSE(25)

___________________________

Iteration 2 [Replaced Standard Scaler with Robust Scaler] [TCN]

Standard scaler would take in outliers at face value and not smooth them , which made the already large variances present between the training and validation and test datasets even larger , this issue was addressed by replacing standardscaler with robustscaler which evened out the outliers thus reducing variances in certain values but not solving overall issues 

No large differences in any values

__________________________

Iteration 3 [Log Transformation] [TCN]

Since it was groups of large values that were causing issues , log transformation was applied which reduced validation and test dataset values a by a good margin , most of train dataset were left intact due to most of it being negative thus skipping log transformation or being very small ,thus reducing loss and bringing MSE values closer to necessary requirement , but a concurrent issue facing all iterations till now has been the steady loss values , instead of dropping as epochs increases which indicates model learning , it is staying constant which means that model is not learning and is just blindly predicting 

Training Loss (0.5)
Validation Loss(1.2)
MSE(6.1)

_____________________________

Switched Model [LSTM]

Since the TCN model was constantly returning large loss and MSE values no matter how standardized and preprocessed the dataset was ,so after going through various research papers , i decided to test out a basic lstm system , since it was much more potent at understanding long sequences of data, after moving to just a basic lstm model, it could predict basic movement of the market but not the magnitude , as it really only understands temporal features it could capture movements but not the magnitude of these moves ,after it predicted the movements to a certain degree i moved onto the scraper 

Training Loss (0.04)
Validation Loss (0.13)
MSE(0.11)

This model worked , and returned a good enough output temporarily leaving it ,to build the rest of the system , will come back add technical indicators and polish it once the entire system is complete 

____________________________

____________________________

Default Model [SCRAPER]

The default model used a beautiful soup model, that initially looked for keywords in queries ,as in searched the links itself for the certain keywords , which didnt work, then switched to google rss to get certain articles or links but since google rss only showed a limited amount of articles we switched from that as well

___________________________

Iteration 1 [SCRAPER]

We swtiched the scraper to instead of checking google rss or we accessed certain economic sites link and searched for the keywords within these site links , which narrowed down the articles it has to parse through and get more results 

__________________________

Iteration 2[SCRAPER]

Beautiful soup only returned the static pages urls and links ,so the links it could acquire were limited since most sites nowadays used script for these links , to access these dynamic pages , we switched from beautiful soup to selenium which can scroll or move through these sites and access all the links present 

__________________________

Iteration 3 [SCRAPER]

Fully utilizing Selenium now , instead of checking individual tags for certain keywords or links to articles , since BERT is a model that benefits from having "junk" fed to it , we instead removed all select loops and just grabbed every href link on all the pages , this returned around 1200 articles ,from economic times and livemint, moneycontrol had an antibot software "akami" that prevented us from scraping it , and since overcoming that obstacle would question the legality and ethics , i opted not to 

__________________________

Iteration 4[SCRAPER,AUTOMATION]

Since BERT requires a large volume of data , i decided to add more sites to scrape for the model , initially i decided to do this manually ,just adding links to url and then building the for loop myself , i realised it would take alot of time , so instead decided to automate the process ,uttilizing two arrays , one for the website links and one for the saved articles, i also added a bert scroller to this t access more links that might not have been previously accessible just through loading javascript using selenium

__________________________

Iteration 5 [SCRAPER,AUTOMATION]

Added a blacklist array which indicates the sites to not scrape, this was done so that bert does not scrape or access junk sites , also added alot of try exception blocks so that program dosent crash when a single site dosent load

This iteration worked , added a few exception blocks here and there, added functions to check browser health , and make the system more redundant

___________________________

___________________________

Default Model [LOADER]

The default model followed a similar architecture to that of the scraper since they both had a similar function , but this one used newpapery3k to parse through the articles whose links and titles were collected and stored in a json file , the loader is built to extract all the content in main articles and store it in a seperate folder as txt files with their respective title names 

__________________________

Iteration 1 [LOADER]

Since newspapery3k class article was constantly reloaded on every scroll but the contents were not stored to a file or array,we lost all the data except the content that was present on the html doc on the last iteration or the last scroll , to fix this we implemented a string that is constantly concatenated on every loop of the scroll ,as in article writes its content into this string before new scroll is executed 

__________________________

Iteration 2[LOADER]

Sinces articles load the same url constantly even after scrolling the page that is loaded is the initial page ,so scrolling becomes irrelevant ,to prevent this we needed to store the links in an array and after every scroll append the new link to this list and have the article url be called afterwards , so that the url can be skipped and article skips all url that came beforehand ,which are the ones present in array 

Scroll-->Click-->Article loaded-->Content parsed-->content stored-->Scroll again
                                |
                                |
                                |
Scroll-->Click-->Url Stored-->Url check runs-->Article loads-->Content parsed-->Content stored-->Scroll again






