import psycopg2
import streamlit as st

def insert_data(values):
    connection = psycopg2.connect(user="postgres",
                                        password="Junijuli1",
                                        host="localhost",
                                        port="5433",
                                        database="sentimen_elte")
    cursor = connection.cursor()
    for ind in values.index:
        try:
            postgres_insert_query = """ INSERT INTO dataset_etle(id, date,user,tweet,translate,textblobsubjectivity,textblobpolarity,sentimen) VALUES (%s, %s,%s,%s, %s, %s, %s, %s)"""
            record_to_insert = (int(values['id'][ind]),values['date'][ind], values['user'][ind], values['tweet'][ind], values['translate'][ind], float(values['TextBlob_Subjectivity'][ind]), float(values['TextBlob_Polarity'][ind]), int(values['sentimen'][ind]))
            cursor.execute(postgres_insert_query, record_to_insert)
            connection.commit()
            count = cursor.rowcount
            print(count, "Record inserted successfully into table")
        except (Exception, psycopg2.Error) as error:
            st.write("Failed to insert record into dataset_etle table", error)
    if connection:
        cursor.close()
        connection.close()
        print("PostgreSQL connection is closed")