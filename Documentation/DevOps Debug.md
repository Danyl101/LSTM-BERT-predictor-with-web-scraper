                                    DEV OPS DEBUG LOG

Tag=[FLASK,SELENIUM]

_____________________________________

Error [FLASK]

URL Not found

 * Running on http://127.0.0.1:5000
 
app = Flask(__name__) #creates flask app instance

@app.route('/scrape', methods=['GET'])

Reason & Solution

So the issue here is that flask only provides the base path and does not provide complete route, as in flask only prints the port or link of base site and you have to add the routes you need to the link by yourself 

http://127.0.0.1:5000/ --> http://127.0.0.1:5000/scrape
        |                                |
        |                                |
    initial                            final

________________________________________________________

Error [SELENIUM]

MaxRetryError
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='localhost', port=59248): Max retries exceeded with url: /session/a24084085cd013ef5d81f7b362375242/source (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x00000272896A5A30>: Failed to establish a new connection: [WinError 10061] No connection could be made because the target machine actively refused it'))

Reason & Solution

The python script tried to run the program on the instantiated chrome server but the chrome driver either got blocked by firewall , shut down after max limit, or terminated manually by driver quit()

Here it was most probably shutting down after max limit

driver.quit()

print(driver.page_source[:2000])

Since the program was trying to reach a driver that no longer existed it just kept running and eventually shut itself down leading to error

_______________________________________

Error [SELENIUM]


Traceback (most recent call last):
  File "d:\Prediction Model\Scrapers\Scraper.py", line 89, in <module>
    cleaned_articles = scrape_multiple_sites(websites, max_scrolls=15)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "d:\Prediction Model\Scrapers\Scraper.py", line 63, in scrape_multiple_sites
    data = scroll_and_scrape(driver, site, max_scrolls=max_scrolls)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "d:\Prediction Model\Scrapers\Scraper.py", line 30, in scroll_and_scrape
    link = elem.get_attribute('href')
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\danyl\AppData\Roaming\Python\Python312\site-packages\selenium\webdriver\remote\webelement.py", line 232, in get_attribute
    attribute_value = self.parent.execute_script(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\danyl\AppData\Roaming\Python\Python312\site-packages\selenium\webdriver\remote\webdriver.py", line 551, in execute_script
    return self.execute(command, {"script": script, "args": converted_args})["value"]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\danyl\AppData\Roaming\Python\Python312\site-packages\selenium\webdriver\remote\webdriver.py", line 454, in execute
    self.error_handler.check_response(response)
  File "C:\Users\danyl\AppData\Roaming\Python\Python312\site-packages\selenium\webdriver\remote\errorhandler.py", line 232, in check_response
    raise exception_class(message, screen, stacktrace)
selenium.common.exceptions.StaleElementReferenceException: Message: stale element reference: stale element not found in the current frame
  (Session info: chrome=138.0.7204.157); For documentation on this error, please visit: https://www.selenium.dev/documentation/webdriver/troubleshooting/errors#staleelementreferenceexception
  
Stacktrace:
        GetHandleVerifier [0x0xac1a33+62339]
        GetHandleVerifier [0x0xac1a74+62404]
        (No symbol) [0x0x902123]
        (No symbol) [0x0x9088d9]
        (No symbol) [0x0x90ae93]
        (No symbol) [0x0x991794]
        (No symbol) [0x0x96f3bc]
        (No symbol) [0x0x9907a3]
        (No symbol) [0x0x96f1b6]
        (No symbol) [0x0x93e7a2]
        (No symbol) [0x0x93f644]
        GetHandleVerifier [0x0xd365c3+2637587]
        GetHandleVerifier [0x0xd319ca+2618138]
        GetHandleVerifier [0x0xae84aa+220666]
        GetHandleVerifier [0x0xad88d8+156200]
        GetHandleVerifier [0x0xadf06d+182717]
        GetHandleVerifier [0x0xac9978+94920]
        GetHandleVerifier [0x0xac9b02+95314]
        GetHandleVerifier [0x0xab4c4a+9626]
        BaseThreadInitThunk [0x0x74c75d49+25]
        RtlInitializeExceptionChain [0x0x770ed1ab+107]
        RtlGetAppContainerNamedObjectPath [0x0x770ed131+561]


Reason & Solution
This issue occurs when the DOM elements or html tags that the system takes is no longer relevant ,as in it takes DOM elements but selenium scrolls through the page ,so these DOM elements that are loaded becomes irrevelant causing this issue

_______________________________







