# Applied-ML
Repository containing all the major applications of Machine Learning / Deep Learning Projects done by me . Every project is sorted by category and followed by a small description of its application features, dataset and methods used.

# Natural Language Processing ( NLP )


 1. ### Named Entity Recognition ( NER ) using Spacy
Named entity Recognition model to extract informationand classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. 


Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/NLP_Named_Entity_Recognition.ipynb)

2. ### Topic Modelling using Genism and NLTK
Using statistical modeling techniques for discovering the abstract “topics” that occur in a collection of documents and thus helping in automatic (unsupervised ) title / classification process. Here we used News Headlines from reputable Australian news source ABC (Australian Broadcasting Corporation) and used LDA, Genism doc2bow for modelling purposes. 

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/NLP_Topic_Modelling.ipynb)

3. ### Sentiment Analysis

Using different State-of-the-art techniques / models for natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. 

i. Determing sentiment polarity for time-series modelling. Using sentiment analysis as a factor and determing how Stocks / demands are affected from people's tweets over twitter with time. Libraries used : NLTK, Flair

Link : [Click Here](https://github.com/dudesparsh/100-days-of-code/blob/master/main.py)

ii. Classifying IMDB reviews ( NLP )

Classifying IMDB reviews as positive or Negative based on the review text using LSTMs ( 92.4 % accuracy ) using fast.ai models ( AWD-LSTM ). Based on transfer learning approach with classificaiton model trained on Wikitext-103 dataset

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/IMDB_NLP.ipynb)

iii. Airline tweet sentiment Analysis ( using Flair ) : Classifying airline passengers tweets for determing their in-flight experience using NLP

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/Sentiment_Analysis_using_Flair.ipynb)

iv. NLTK sentiment Analysis :  [Click Here](https://github.com/dudesparsh/100-days-of-code/blob/master/NLTK_Sentiment_Analysis.ipynb)

v. TextBlob sentiment Analysis :  [Click Here](https://github.com/dudesparsh/100-days-of-code/blob/master/TextBlob_Sentiment_Analysis.ipynb)
 


4. ### Zero shot classificaiton using Hugging Face Transformers

Using SOTA transformers and hugging face model to perform zero-shot classification or doing unsupervised modelling for text classification. Here determing product categories from unstructured dataset.

Link : [Click here](https://github.com/dudesparsh/Applied-ML/blob/master/Product_categories_from_unstructured_dataset.ipynb)

5. ### Document similarity

Modelling resume and job description similarity for determining relavancy score and how close resume handles what's required in job. Simple model using cosine similarity ( to be continued )

Link : [Click here](https://github.com/dudesparsh/Applied-ML/blob/master/Resume_Scorer.ipynb)


# Computer Vision

 1. ### Finger Detection and Counting Using OpenCV
The program can detect hand from live video and through OpenCV and Convell Hull technique counts the number of fingers shown to camera.

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/Finger_Count.ipynb)

2. YOLO Application
3. ### Number Plate Blurring using HaarCascade in OpenCV
The program can automatically detect the Russian car number plate present in the image and Apply blurring to the image.

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/Number_Plate_Blurring_using_HaarCascade_%28OpenCV%29.ipynb)

4. ### Keras MNIST Classfication
Classfying the MNIST Fashion dataset ( 60000 Images ) into different categories by CNN model in Keras.

Link : [Click here](https://github.com/dudesparsh/100-days-of-code/blob/master/Keras%20Fashion%20MNIST%20Image%20Classification.ipynb)


5. ### Image Processing using OpenCV
Applying basic image processing techniques such as : Sobel Edge detection, Thresholding, kernels & blending using OpenCV

Link : [Click here](https://github.com/dudesparsh/100-days-of-code/blob/master/07_Image_Processing_Assessment.ipynb)


6. ### Optical Character Recognition ( OCR )
Using state of the art Tesseract OCR for extracting data from images.

Link : [Click here](https://github.com/dudesparsh/Applied-ML/blob/master/Tesseract_OCR.ipynb)

# Python / Data Science projects

 1. ### RSS Feed parser
Using the feed parser library to extract RSS feeds for different podcasts and inserting the data into database into PostGRE's database. Hence automating the feed parsing for simple use in apps / website feeds

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/rss_feed.py)

2. ### APPLE stock price prediction using LSTM
Time series modelling for APPLE stocks. Using past data to predict the future price of APPLE stocks using LSTM models.

Link : [Click Here](https://github.com/dudesparsh/100-days-of-code/blob/master/APPLE_stock_price_prediction_using_LSTM.ipynb)

3. ### Customer Value Prediction

Using data science / machine learning models for prediction of the net profit attributed to the entire future relationship with a customer (Customer Life time Value).

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/Customer_Value_Prediction.ipynb)

4. ### Text to Speech System

Using Google's SOTA speech recognition library to automatically convert speect into text

Link : [Click here](https://github.com/dudesparsh/100-days-of-code/blob/master/Text_to_speech_Demo.ipynb)


# Fast.ai
1. ### Classifying IMDB reviews ( NLP )
Classifying IMDB reviews as positive or Negative based on the review text using LSTMs ( 92.4 % accuracy ) using fast.ai models ( AWD-LSTM ).

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/IMDB_NLP.ipynb)

2.  ### Collaborative Filtering on MovieLens Dataset
Using collaborative filtering to sort out the Top movies from the MovieLens dataset ( containing 100k data )

Link : [Click here](https://github.com/dudesparsh/Applied-ML/blob/master/Movielens_Collaborative_Filtering.ipynb)

3. ### Camvid Image Segmentation
Using fastai vision to segment the images using ResNet34 CNN model ( Accuracy : 92.5% ) on Camvid Data

Link : [Click here](https://github.com/dudesparsh/Applied-ML/blob/master/Camvid.ipynb)

4. ### Regression with BIWI Head Pose
Using fastai to find the center of the face in a head pose using ResNet34 CNN model

Link : [Click here](https://github.com/dudesparsh/Applied-ML/blob/master/Regression_with_BIWI_head_pose.ipynb)

5. ### Custom Bear Dataset Classification
Script based extraction of data from google images & Application of ResNet34 & ResNet50 CNN model to classify the images in custom bears dataset using fastai.

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/Custom%20dataset%20Classification.ipynb)




# Web Scrapers
1. ### Immo Scout ( end-to-end )
Using BeautifulSoup to scrape data from Biggest German Real estate site and storing it in remote SQL database for analytics purposes.

Link : [Click Here](https://github.com/dudesparsh/Applied-ML/blob/master/Immo_Deploy_Final_(_Updated_).ipynb)

2.  ### Food.com 
Using BeautifulSoup to scrape Food.com ( Food.com is a digital brand and online social networking service featuring recipes from home cooks and celebrity chefs, food news, new and classic shows, and pop culture )

Link : [Click here](https://github.com/dudesparsh/100-days-of-code/blob/master/Food_scraper_updated.ipynb)

3. ### Porsche Scraper
Extracting cars inventory data from Porsche.com.

Link : [Click here](https://github.com/dudesparsh/100-days-of-code/blob/master/Porsche_Scraper.ipynb)

4. ### Capterra Scraper 
Using bs4 for scraping softwares data from Capterra website

Link : [Click here](https://github.com/dudesparsh/100-days-of-code/blob/master/Capterra_Scraper.ipynb)

5. ### Clinic Portal Scraper 
Scraping script for extracting data of all clinics from the portal

Link : [Click Here](https://github.com/dudesparsh/100-days-of-code/blob/master/Clinic_Portal_Scraper.ipynb)
