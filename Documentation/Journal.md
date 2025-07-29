                                    POST CONSTRUCTION MODEL LOG


Tags:[LSTM,TCN,SCRAPER,AUTOMATION,LOADER,SEPERATOR,BILSTM]
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

Training Loss (0.5)
Validation Loss (0.8)
MSE(0.11)

This model worked , and returned a good enough output temporarily leaving it ,to build the rest of the system , will come back add technical indicators and polish it once the entire system is complete 

____________________________

Iteration 1 [BILSTM]

Decided to upgrade the lstm to a bilstm , since the bilstm would be able to capture much more temporal dependencies being of capable understanding both previous and future patterns from a certain point , also since training and validation,test having such a different values was a consistent issue , decided to add a custom loss function to the training dataset while keeeping original loss function for validationa and test

Training Loss (0.2)
Validation Loss(0.5)
MSE(0.05)

____________________________

Iteration 2 [BILSTM]

So there was an apparent issue with close being fed into validation and test dataset as well , which lead to a small data leakage , which skewed the results of the final model , after correcting this the results of final model were and added other metrics along mse for various different cases ,

Training Loss (0.2)
Validation Loss(0.5)
MSE: 0.0727, RMSE: 0.2696, MAE: 0.2419, R²: -0.5629, MAPE: 9.99%

__________________________

Iteration 3 [BILSTM]

Since the model now had somewhat good results and the overall architecture was good , i decided to begin fine tuning the hyperparameters , for this i implemented bayesian optimization ,initially wanted to do gaussian functions by myself but since that requires some time and acquisition function could be difficult ,i decided to go for optuna , but there was an issue where optuna constantly became stuck ,so added in a bunch of try catch blocks everywhere and added return(inf) to the optuna block which solved it


__________________________
__________________________

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

Iteration 4[SCRAPER]

Since BERT requires a large volume of data , i decided to add more sites to scrape for the model , initially i decided to do this manually ,just adding links to url and then building the for loop myself , i realised it would take alot of time , so instead decided to automate the process ,uttilizing two arrays , one for the website links and one for the saved articles, i also added a bert scroller to this t access more links that might not have been previously accessible just through loading javascript using selenium

__________________________

Iteration 5 [SCRAPER]

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

___________________________

Iteartion 3 [LOADER]

Build an advanced get function which creats a link with the site and receives logs from there , thus allowing more advanced debugging, added debugging into every single layer of the program and found the core issues which was a pathing issue , where new spaces caused error in the windows file naming system , updated the sanitizer of filenames  

___________________________

Iteration 4 [LOADER]

Ran into issues where some sites where parsed properly and some werent ,this was caused by the issue that most sites dont use proper tags and instead just use " " and strong tags inside html for paragraphs , which breaks most libraries used for parsing , created a seperate function for dealing with such raw html tags by utilizing xpath 

___________________________

Iteration 5 [LOADER]

Abandoned the xpath function and instead decided to use a new library playwright , which is capable of rendering and parsing even the most js heavy and complicated and with a few tweaks , it was capable of parsing through the most broken html sites , but since it was somewhat slow it is used alongside selenium-newspaper,as in certain links are fed into playwright and rest are fed into newspaper
___________________________

Iteration 6 [LOADER]

So playwright began working for me , but certain articles became stuck on parsing for extremely long times ,so added a timeout function for playwright , also split everything into modules and its own seperate programs, there was a tiny issue with the loader where if the link of the site wasnt explicitly named then it would automatically go to selenium-newspaper which could cause issues , so instead of checking site name at loader , every link is sent to selenium , and if it fails then its sent to playwright

                                            Sucess
            LOADER---->SELENIUM-NEWSPAPER------------>FILE SAVED
                            |
                            | Fail
                            |
                        PLAYWRIGHT----->FILE SAVED
___________________________

Default Model [SEPARATOR]

So only a few files were actually properly extracted by loader , the exact reason or issue for others not being extracted is still not known due to incomplete logging(yet to finish),most possibly due to pay walls and sleep times being too little(2-3),but since a universal loader is extremely difficult , i made a system to seperate the scraped txts and links from the ones yet to be scraped , which is the separator , it identifies the file names of txt files loaded in bert content folder and matches that with the article titles present in dataset, and remove both of these to a new file thus preserving data , while making it usable 

___________________________

Iteration 1 [SEPARATOR]

So the files had a small pathing issue which arose from the fact that the relative path of the files were off,since the path of the actual program being executed was different , so the actual files existed in a path that was completely different , than the filepaths the program created due to this relative directory issue 








