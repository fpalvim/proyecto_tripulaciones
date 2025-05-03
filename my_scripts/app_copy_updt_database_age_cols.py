import psycopg2
import pandas as pd
from datetime import datetime
from IPython.display import HTML, display
import requests
import re
import json
from dotenv import load_dotenv
import os
import ast  # Import the ast module

load_dotenv()

RENDER_DB_URI = os.getenv("RENDER_DB_URI")
TABLE_NAME = 'all_events'
UNIQUE_ID_COLUMN = 'event_id'
LIST_COLUMNS_TO_CONVERT = ['tags']

# --- Function to safely parse a list ---
def safe_parse_list(tag_str):
    try:
        return ast.literal_eval(tag_str) if isinstance(tag_str, str) else tag_str
    except:
        return []

# --- English to Spanish translation mapping ---
english_tags = [
    'Audio', 'Baby', 'Babygroup', 'Bookshop', 'Children', 'Children & Youth',
    'Childrenandfamilies', 'Childrens', 'Class, Training, or Workshop', 'Community',
    'Concert or Performance', 'Conference', 'Event', 'Family', 'Family & Education',
    'Familyworkshop', 'Gathering', 'Humour', 'Kids', 'Lettering', 'Parenting',
    'Podcast', 'Radio', 'Standup', 'Standupcomedy', 'Storyteller', 'Storytelling',
    'Workshop', 'kids_events'
]

spanish_tags = [
    'Audio', 'Bebé', 'Grupo de bebés', 'Librería', 'Niños', 'Niños y jóvenes',
    'Niños y familias', 'Niños', 'Clase, formación o taller', 'Comunidad',
    'Concierto o actuación', 'Conferencia', 'Evento', 'Familia', 'Familia y educación',
    'Taller familiar', 'Reunión', 'Humor', 'Niños', 'Lettering', 'Crianza',
    'Podcast', 'Radio', 'Monólogo', 'Comedia en Barcelona', 'Narrador',
    'Narración de cuentos', 'Taller', 'eventos_infantiles'
]

# --- Create translation dictionary ---
translation_dict = dict(zip(english_tags, spanish_tags))

# --- Function to extract age information ---
def extract_age_final(row):
    text = f"{row['name']} {row['summary']}" if pd.notna(row['summary']) else row['name']

    all_ranges_str = []
    min_ages = []
    max_ages = []

    # Buscar múltiples rangos en la misma línea (ej: 11 y 12 años)
    multi_range_patterns = re.findall(r'(\d+)\s*y\s*(\d+)\s*(?:years?|year?|anos?|año|años)\b', text, re.IGNORECASE)
    for min_age, max_age in multi_range_patterns:
        all_ranges_str.append(f"{min_age}-{max_age}")
        min_ages.append(int(min_age))
        max_ages.append(int(max_age))

    # Buscar rangos con guion o "a" (ej: 6-12 años)
    range_patterns = re.findall(r'(\d+)\s*(?:-|a)\s*(\d+)\s*(?:years?|year?|anos?|año|años)\b', text, re.IGNORECASE)
    for min_age, max_age in range_patterns:
        all_ranges_str.append(f"{min_age}-{max_age}")
        min_ages.append(int(min_age))
        max_ages.append(int(max_age))

    # Buscar patrones "A partir de"
    apartir_patterns = re.findall(r'A partir de\s*(\d+)\s*(?:years?|year?|anos?|año|años)?\b', text, re.IGNORECASE)
    for age in apartir_patterns:
        all_ranges_str.append(f"+{age}")
        min_ages.append(int(age))
        max_ages.append(14)  # Establecemos edad_max a 14

    # Buscar patrones con "+"
    plus_patterns = re.findall(r'\+(\d+)\s*(?:years?|year?|anos?|año|años)?\b', text, re.IGNORECASE)
    for age in plus_patterns:
        all_ranges_str.append(f"+{age}")
        min_ages.append(int(age))
        max_ages.append(14)  # Establecemos edad_max a 14

    # Buscar edades individuales
    single_patterns = re.findall(r'(\d+)\s*(?:years?|year?|anos?|año|años)\b', text, re.IGNORECASE)
    extracted_single = set()
    for age_range in all_ranges_str:
        if '-' in age_range:
            extracted_single.add(age_range.split('-')[0])
            extracted_single.add(age_range.split('-')[1])
        elif '+' in age_range:
            extracted_single.add(age_range[1:])
        else:
            extracted_single.add(age_range)

    for age_str in single_patterns:
        if age_str not in extracted_single:
            all_ranges_str.append(age_str)
            min_ages.append(int(age_str))
            max_ages.append(int(age_str))

    age_range_col = ' | '.join(all_ranges_str) if all_ranges_str else None
    final_min_age = min(min_ages) if min_ages else None
    final_max_age = max(max_ages) if max_ages else None

    return pd.Series({'age_range': age_range_col, 'min_age': final_min_age, 'max_age': final_max_age})

# --- Extract data from Eventbrite ---
url = "https://www.eventbrite.es/d/spain/all-events/?subcategories=15003%2C15004%2C15005%2C15006&page=1"

headers = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9,it-IT;q=0.8,it;q=0.7,pt;q=0.6,es;q=0.5",
    "priority": "u=1, i",
    "referer": "https://www.eventbrite.es/d/spain/all-events/?subcategories=15003%2C15004%2C15005%2C15006&page=1",
    "sec-ch-ua": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
    "x-csrftoken": "6a8e093a186e11f0b3df7333c3540c62",
    "x-requested-with": "XMLHttpRequest"
}

cookies = {
    "mgrefby": "https://www.google.com/",
    "guest": "identifier=75b1ab71-48c3-4d01-ad8a-782d93c1f5b6&a=13e8&s=3411e4860099bda2b7af7214fd17839fbe2e76413c4fea6258e306e45ffcc717",
    "G": "v=2&i=75b1ab71-48c3-4d01-ad8a-782d93c1f5b6&a=13e8&s=b30ebeb1875f2120a85bd6000eefa890f9f4cf55",
    "csrftoken": "6a8e093a186e11f0b3df7333c3540c62",
    "session": "identifier=822257312e8c43ce8e9bec649fedd725&issuedTs=1744553017&originalTs=1744552296&s=1815426704c353afae38072c44048b429f12aa1d9f323757b7e9fbecd97c17ff"
}

response = requests.get(url, headers=headers, cookies=cookies)

if response.status_code == 200:
    data = response.json()
    search_data = data.get("search_data")
    if search_data and "events" in search_data:
        events = search_data["events"].get("results", [])
        extracted_data = []
        for event in events:
            start_time_str = event.get("start_date") + " " + event.get("start_time") if event.get("start_date") and event.get("start_time") else None
            end_time_str = event.get("end_date") + " " + event.get("end_time") if event.get("end_date") and event.get("end_time") else None
            duration = None
            image_url = event.get("image", {}).get("url")
            venue = event.get("primary_venue", {})
            address = venue.get("address", {})
            longitude = float(address.get("longitude")) if address.get("longitude") else None
            latitude = float(address.get("latitude")) if address.get("latitude") else None

            if start_time_str and end_time_str:
                try:
                    start_datetime = datetime.fromisoformat(start_time_str)
                    end_datetime = datetime.fromisoformat(end_time_str)
                    duration = str(end_datetime - start_datetime)
                except ValueError:
                    duration = "Invalid date/time format"

            item = {
                "name": event.get("name"),
                "url": event.get("url"),
                "start_time": event.get("start_time"),
                "start_date": event.get("start_date"),
                "end_time": event.get("end_time"),
                "end_date": event.get("end_date"),
                "duration": duration,
                "venue_name": event.get("primary_venue", {}).get("name"),
                "venue_address": event.get("primary_venue", {}).get("address", {}).get("localized_address_display"),
                "longitude": longitude,
                "latitude": latitude,
                "summary": event.get("summary"),
                "is_online_event": event.get("is_online_event"),
                "tickets_url": event.get("tickets_url"),
                "tags": [tag.get("display_name") for tag in event.get("tags", [])],
                "event_id": event.get("id"),
                "data_source": "eventbrite"
            }
            extracted_data.append(item)

        # Create DataFrame from extracted data
        eventbrite_df = pd.DataFrame(extracted_data)

        # --- Translate tags to Spanish for Eventbrite ---
        def safe_parse_list(tag_str):
            try:
                return ast.literal_eval(tag_str) if isinstance(tag_str, str) else tag_str
            except:
                return []

        # Step 1: Make sure 'tags' are proper lists (not strings)
        eventbrite_df['tags'] = eventbrite_df['tags'].apply(safe_parse_list)

        # Step 2: Translate tags for 'eventbrite' data_source rows
        def translate_tag_list(tag_list):
            return [translation_dict.get(tag.strip(), tag.strip()) for tag in tag_list]

        eventbrite_df.loc[eventbrite_df['data_source'] == 'eventbrite', 'tags'] = (
            eventbrite_df.loc[eventbrite_df['data_source'] == 'eventbrite', 'tags'].apply(translate_tag_list)
        )
    else:
        print("No events data found in Eventbrite response.")
        eventbrite_df = pd.DataFrame() # Initialize an empty DataFrame
else:
    print(f"Failed to fetch Eventbrite data. Status code: {response.status_code}")
    eventbrite_df = pd.DataFrame() # Initialize an empty DataFrame

# Extraer datos ayuntamiento de Madrid

# descargando los datos de Agenda de actividades y eventos de la API del Portal de datos abiertos del Ayuntamiento de Madrid

url = "https://datos.madrid.es/egob/catalogo/300107-0-agenda-actividades-eventos.json"

try:
    response = requests.get(url)
    response.raise_for_status()
    raw_text = response.text
    cleaned_text = re.sub(r"\\(?![nrtbf\"\\/])", "", raw_text)
    data = json.loads(cleaned_text)

    if isinstance(data, dict):
        print(f"Top-level dictionary keys: {data.keys()}")
        if '@graph' in data and isinstance(data['@graph'], list):
            df = pd.DataFrame(data['@graph'])
            print(df.head())
        else:
            print("The list of events was not found within the '@graph' key.")
            df = pd.DataFrame() # Initialize an empty DataFrame
    else:
        print("Parsed JSON is not a dictionary as expected.")
        df = pd.DataFrame() # Initialize an empty DataFrame

except requests.exceptions.RequestException as e:
    print(f"Error fetching data from Ayuntamiento de Madrid: {e}")
    df = pd.DataFrame() # Initialize an empty DataFrame
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from Ayuntamiento de Madrid: {e}")
    df = pd.DataFrame() # Initialize an empty DataFrame
except Exception as e:
    print(f"An unexpected error occurred while fetching Ayuntamiento de Madrid data: {e}")
    df = pd.DataFrame() # Initialize an empty DataFrame

if not df.empty:
    # Haciendo el filtrado del dataset
    filtered_df = df[df['audience'].isin(['Familias', 'Niños', 'Jovenes,Niños', 'Niños,Familias', 'Jovenes,Niños,Familias'])].copy() # Use .copy() to avoid SettingWithCopyWarning

    extracted_data = []
    for index, row in filtered_df.iterrows():
        extracted_event = {
            "name": row["title"],
            "url": row["link"],
            "image": None,
            "start_time": row["time"],
            "start_date": pd.to_datetime(row["dtstart"]).date() if pd.notna(row["dtstart"]) else None,
            "end_time": None,
            "end_date": pd.to_datetime(row["dtend"]).date() if pd.notna(row["dtend"]) else None,
            "duration": None,
            "venue_name": row["event-location"],
            "venue_address": row["address"].get("area", {}).get("street-address")
                                    if isinstance(row["address"], dict) and isinstance(row["address"].get("area"), dict) else None,
            "longitude": row["location"].get("longitude")
                                    if isinstance(row["location"], dict) and row["location"].get("longitude") else None, # Comprueba si la ubicación es un diccionario
            "latitude": row["location"].get("latitude")
                                    if isinstance(row["location"], dict) and row["location"].get("latitude") else None, # Comprueba si la ubicación es un diccionario
            "summary": row["description"],
            "is_online_event": None,
            "tickets_url": row["link"],
            "tags": [row["audience"]] if pd.notna(row["audience"]) else [],
            "event_id": row["uid"],
            "data_source": "ayuntamiento madrid"
        }
        extracted_data.append(extracted_event)

    ayuntamiento_madrid_df = pd.DataFrame(extracted_data)
    print(ayuntamiento_madrid_df.head())
    print(ayuntamiento_madrid_df.info())
else:
    ayuntamiento_madrid_df = pd.DataFrame() # Initialize an empty DataFrame

# Concatenando los datos extraidos

# Creando una lista con los datasets existentes
dataframes_list = [eventbrite_df, ayuntamiento_madrid_df]

# Concatenando los dataframes
combined_df = pd.concat(dataframes_list, ignore_index=True)

# Applying the age extraction function to the combined DataFrame
combined_df[['age_range', 'min_age', 'max_age']] = combined_df.apply(extract_age_final, axis=1)

combined_df.to_csv('data/raw/combined_df.csv',index=False)


def update_event_database_render(current_events_df):
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(RENDER_DB_URI)
        cursor = conn.cursor()

        # Add the new columns if they don't exist
        cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS age_range TEXT")
        cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS min_age INTEGER")
        cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS max_age INTEGER")
        conn.commit()
        print(f"Table {TABLE_NAME} altered to add new columns (if they didn't exist).")

        base_columns = [col for col in current_events_df.columns if col != UNIQUE_ID_COLUMN and col not in ['age_range', 'min_age', 'max_age']]
        expected_columns = [UNIQUE_ID_COLUMN] + base_columns + ['age_range', 'min_age', 'max_age']
        placeholders = ', '.join(['%s'] * len(expected_columns))
        insert_sql = f"INSERT INTO {TABLE_NAME} ({', '.join(expected_columns)}) VALUES ({placeholders})"

        cursor.execute(f"SELECT {UNIQUE_ID_COLUMN} FROM {TABLE_NAME}")
        existing_ids = {row[0] for row in cursor.fetchall()}

        new_events_to_insert = []
        for index, row in current_events_df.iterrows():
            unique_id = row[UNIQUE_ID_COLUMN]
            if unique_id not in existing_ids:
                data_to_insert = [unique_id]
                for col in base_columns:
                    value = row[col]
                    if isinstance(value, HTML):
                        data_to_insert.append(str(value))
                    elif isinstance(value, list):
                        data_to_insert.append(','.join(map(str, value)))
                    else:
                        data_to_insert.append(value)

                # Handle potential None values for min_age and max_age
                min_age_val = row['min_age'] if pd.notna(row['min_age']) else None
                max_age_val = row['max_age'] if pd.notna(row['max_age']) else None

                data_to_insert.extend([row['age_range'], min_age_val, max_age_val])
                new_events_to_insert.append(tuple(data_to_insert))

        if new_events_to_insert:
            cursor.executemany(insert_sql, new_events_to_insert)
            conn.commit()
            print(f"Inserted {len(new_events_to_insert)} new events into the Render database.")
        else:
            print("No new events found.")

    except psycopg2.Error as e:
        print(f"Error interacting with the PostgreSQL database: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def load_all_events_from_db_render():
    conn = None
    all_events_df = None
    try:
        conn = psycopg2.connect(RENDER_DB_URI)
        all_events_df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    except psycopg2.Error as e:
        print(f"Error loading data from the PostgreSQL database: {e}")
    finally:
        if conn:
            conn.close()
    return all_events_df



# --- Execute the data extraction and database update ---
dataframes_list = [eventbrite_df, ayuntamiento_madrid_df]
combined_df = pd.concat(dataframes_list, ignore_index=True)

# Apply the age extraction function
combined_df[['age_range', 'min_age', 'max_age']] = combined_df.apply(extract_age_final, axis=1)

print("First few rows of combined_df with age columns:")
print(combined_df[['name', 'summary', 'age_range', 'min_age', 'max_age']].head())

update_event_database_render(combined_df)

all_events_df_render = load_all_events_from_db_render()
if all_events_df_render is not None:
    print("\nAll events loaded from the Render database:")
    print(all_events_df_render[['name', 'age_range', 'min_age', 'max_age']].head())