import feedparser
import pandas as pd
from pprint import pprint
import psycopg2
import json
#url = 'https://defenseofthepatience.libsyn.com/rss'

#Getting feeds from given url using feedparser library
def parse(url):
    return feedparser.parse(url)


def get_source(parsed):
    feed = parsed['feed']
    return {
        
        'title':feed['title'],
        'content':feed['content'],
        'published':feed['published'],
        'summary':feed['summary'],
        'author':feed['author'],
        'link':feed['link'],
                
    }
    
#Extracting the information from the entries ( list of all posts )
#Summary is same as subtitle and also subtitle is in better formatting
#Hence using subtitle here    
def get_articles(parsed):
    articles = []
    entries = parsed['entries']
    for entry in entries:
        articles.append({
            
            'id':entry['id'],
            'title':entry['title'],
            'link':entry['link'],
#            'summary':entry['summary'],            
            'subtitle':entry['subtitle'],
            'published':entry['published'],
                        
        })
    return articles
#Establishing connection with PostgreSQL database
def get_connection(file):
    
    with open(file) as inFile:
        creds = json.load(inFile)
        
    database = creds['database']
    user = creds['user']
    password = creds['password']
    host = creds['host']
    port = creds['port']
    
    connection = psycopg2.connect(database = database, user=user, password = password, host = host, port = port)
    return connection
    
#Function to save the articles to the PostgreSQL databse

def save_function(result):
    
    #result.to_csv(title_url+'.csv')
    try:
        #Setting connection with the database
        connection = get_connection('connection.json')

        cursor = connection.cursor()
        #Insert data into table only if its id doesn't exist
        postgres_insert_query = """INSERT INTO defence_pat.posts (id, title, link, subtitle, pubdate) VALUES(%s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING;"""
        total = 0
        #Iterating each row of our dataframe
        for index, row in result.iterrows():
            #Format for taking and inserting records in table
            #row['id'] row['title'] row['link'] rows['subtitle'] row['published']
            
            record_to_insert = (row[0], row[1], row[2], row[3], row[4])
            #Inserting new data into database
            cursor.execute(postgres_insert_query, record_to_insert)
            connection.commit()
            count = cursor.rowcount
            print(count, " Record inserted successfully into defense of patience table")
            #total is used to keep the count of total queries
            total = total+1
    #Handling errors        
    except(Exception, psycopg2.Error) as error:
        if(connection):
            print("Failed to insert record into defense of patience table", error)
    finally:
        #closing the database connection
        if(connection):
            print(total, " Queries")
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")

def main():
    #Initialising the feed parser
    print('Started')
    url = 'https://defenseofthepatience.libsyn.com/rss'
    title_url = 'Defence_of_patience'
    feed = parse(url)
    articles = get_articles(feed)
    #Changing of articles as pandas dataframe
    result = pd.DataFrame(articles)
    #If we wish to save data as csv file, use the below function
    #Also modify the save_function parameters
    #save_function(result, title_url)
    save_function(result)
    
    
if __name__ == "__main__":
    main()
